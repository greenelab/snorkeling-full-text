# Global Imports for Positive (1) Negative (0) and ABSTAIN (-1)
from ..global_helpers import (
    ABSTAIN,
    NEGATIVE,
    POSITIVE,
    ltp,
    create_dep_mapper,
    match_dep_path,
    get_members,
    get_tagged_text,
    get_text_in_windows,
)
from collections import OrderedDict
import pathlib

import numpy as np
import pandas as pd
import re


from snorkel.labeling import labeling_function

"""
DISTANT SUPERVISION
"""
path = (
    pathlib.Path(__file__)
    .joinpath("../../knowledge_bases/compound_treats_disease.tsv.xz")
    .resolve()
)
print(path)
pair_df = pd.read_table(path, dtype={"sources": str})
knowledge_base = set()
for row in pair_df.itertuples():
    if not row.sources or pd.isnull(row.sources):
        continue
    for source in row.sources.split("|"):
        source = re.sub(r" \(\w+\)", "", source)
        key = row.drugbank_id, row.doid_id, source
        knowledge_base.add(key)


@labeling_function()
def LF_HETNET_PHARMACOTHERAPYDB(c):
    return (
        POSITIVE
        if (c.compound_cid, c.disease_cid, "pharmacotherapydb") in knowledge_base
        else ABSTAIN
    )


@labeling_function()
def LF_HETNET_CD_ABSENT(c):
    """
    This label function fires -1 if the given Disease Gene pair does not appear
    in the databases above.
    """
    return ABSTAIN if any([LF_HETNET_PHARMACOTHERAPYDB(c) == POSITIVE]) else NEGATIVE


disease_normalization_df = pd.read_csv(
    "https://raw.githubusercontent.com/dhimmel/disease-ontology/"
    "052ffcc960f5897a0575f5feff904ca84b7d2c1d/data/slim-terms-prop.tsv",
    sep="\t",
)


@labeling_function()
def LF_CD_CHECK_DISEASE_TAG(c):
    """
    This label function is used for labeling each passed candidate as either pos or neg.
    Keyword Args:
    c- the candidate object to be passed in.
    """
    disease_name = " ".join(c.word[c.disease_start : c.disease_end])

    # If abbreviation skip since no means of easy resolution
    if len(disease_name) <= 5 and disease_name.isupper():
        return ABSTAIN

    disease_id = c.disease_cid
    result = disease_normalization_df[
        disease_normalization_df["subsumed_name"].str.contains(
            disease_name.lower(), regex=False
        )
    ]
    # If no match then return NEGATIVE
    if result.empty:

        # check the reverse direction e.g. carcinoma lung -> lung carcinoma
        disease_name_tokens = disease_name.split(" ")
        if len(disease_name_tokens) == 2:
            result = disease_normalization_df[
                disease_normalization_df["subsumed_name"].str.contains(
                    " ".join(disease_name_tokens[-1 :: 0 - 1]).lower(), regex=False
                )
            ]

            # if reversing doesn't work then output -t
            if not result.empty:
                slim_id = result["slim_id"].values[0]
                if slim_id == disease_id:
                    return ABSTAIN
        return NEGATIVE
    else:
        # If it can be normalized return ABSTAIN else -1
        slim_id = result["slim_id"].values[0]
        if slim_id == disease_id:
            return ABSTAIN
        else:
            return NEGATIVE


"""
SENTENCE PATTERN MATCHING
"""

treat_indication = {
    "was administered",
    "treated with",
    "treatment of",
    "treatment for",
    "feeding a comparable dose.*of",
    "was given",
    "were injected",
    "be administered",
    "treatment with",
    "to treat",
    "induced regression of",
    "therapy",
    "treatment in active",
    "chemotherapeutic( effect of )?",
    "protection from",
    "promising drug for treatment-refractory",
    "inhibit the proliferation of",
    "monotherapy",
    "treatment of",
}

incorrect_depression_indication = {
    "sharp",
    "produced a",
    "opiate",
    "excitability",
    "produced by",
    "causes a",
    "was",
    "effects of",
    "long lasting {{B}} of",
    "led to a",
}

weak_treatment_indications = {
    "responds to",
    "effective with",
    "inhibition of",
    "hypotensive efficacy of",
    "was effecitve in",
    "treatment",
    "render the disease sero-negative",
    "enhance the activity of",
    "blocked in",
    "(possible|potential) benefits of",
    "antitumor activity",
    "prolactin response",
}

incorrect_compound_indications = {
    "human homolog",
    "levels",
    "pathway",
    "receptor(-subtype)?",
}

palliates_indication = {
    "prophylatic effect",
    "supression of",
    "significant reduction in",
    "therapy in",
    "inhibited",
    "prevents or retards",
    "pulmonary vasodilators",
    "management of",
    "was controlled with",
}

compound_indications = {
    "depleting agents",
    "blocking agents",
    "antagonist",
    "antirheumatric drugs",
    "agonist",
}

trial_indications = {
    "clinical trial",
    "(controlled, )?single-blind trial",
    "double-blind.+trial",
    "multi-centre trial",
    "double-blind",
    "placebo",
    "trial(s)?",
    "randomized",
}


@labeling_function()
def LF_CtD_TREATS(c):
    window_text = get_text_in_windows(c)

    compound_left_window = " ".join(window_text["left_window"][0])
    compound_right_window = " ".join(window_text["right_window"][0])

    if any(
        [
            re.search(ltp(treat_indication), window_text["text_between"], flags=re.I),
            re.search(ltp(treat_indication), compound_left_window, flags=re.I),
            re.search(ltp(treat_indication), compound_right_window, flags=re.I),
        ]
    ):
        return POSITIVE

    return ABSTAIN


@labeling_function()
def LF_CD_CHECK_DEPRESSION_USAGE(c):
    window_text = get_text_in_windows(c)
    disease_left_window = " ".join(window_text["left_window"][1])
    disease_right_window = " ".join(window_text["right_window"][1])

    entity_start = window_text["entity_columns"]["entity_two_start"]
    entity_end = window_text["entity_columns"]["entity_two_end"]
    if "depress" in " ".join(c.word[c[entity_start] : c[entity_end]]):
        if any(
            [
                re.search(
                    ltp(incorrect_depression_indication),
                    disease_left_window,
                    flags=re.I,
                ),
                re.search(
                    ltp(incorrect_depression_indication),
                    disease_right_window,
                    flags=re.I,
                ),
            ]
        ):
            return NEGATIVE

    return ABSTAIN


@labeling_function()
def LF_CtD_WEAKLY_TREATS(c):
    """
    This label function is designed to look for phrases
    that have a weak implication towards a compound treating a disease
    """
    window_text = get_text_in_windows(c)
    compound_left_window = " ".join(window_text["left_window"][0])
    compound_right_window = " ".join(window_text["right_window"][0])

    if any(
        [
            re.search(
                ltp(weak_treatment_indications),
                " ".join(window_text["text_between"]),
                flags=re.I,
            ),
            re.search(
                ltp(weak_treatment_indications), compound_left_window, flags=re.I
            ),
            re.search(
                ltp(weak_treatment_indications), compound_right_window, flags=re.I
            ),
        ]
    ):
        return POSITIVE

    return ABSTAIN


@labeling_function()
def LF_CD_INCORRECT_COMPOUND(c):
    """
    This label function is designed to capture phrases
    that indicate the mentioned compound is a protein not a drug
    """
    window_text = get_text_in_windows(c)
    compound_left_window = " ".join(window_text["left_window"][0])
    compound_right_window = " ".join(window_text["right_window"][0])

    if any(
        [
            re.search(
                ltp(incorrect_compound_indications),
                window_text["text_between"],
                flags=re.I,
            ),
            re.search(
                ltp(incorrect_compound_indications), compound_left_window, flags=re.I
            ),
            re.search(
                ltp(incorrect_compound_indications), compound_right_window, flags=re.I
            ),
        ]
    ):
        return NEGATIVE

    return ABSTAIN


@labeling_function()
def LF_CpD_PALLIATES(c):
    """
    This label function is designed to look for phrases
    that could imply a compound binding to a gene/protein
    """
    window_text = get_text_in_windows(c)
    compound_left_window = " ".join(window_text["left_window"][0])
    compound_right_window = " ".join(window_text["right_window"][0])

    if any(
        [
            re.search(
                ltp(palliates_indication), window_text["text_between"], flags=re.I
            ),
            re.search(ltp(palliates_indication), compound_left_window, flags=re.I),
            re.search(ltp(palliates_indication), compound_right_window, flags=re.I),
        ]
    ):
        return POSITIVE

    return ABSTAIN


@labeling_function()
def LF_CtD_COMPOUND_INDICATION(c):
    """
    This label function is designed to look for phrases
    that implies a compound increaseing activity of a gene/protein
    """
    window_text = get_text_in_windows(c)
    compound_left_window = " ".join(window_text["left_window"][0])
    compound_right_window = " ".join(window_text["right_window"][0])

    if any(
        [
            re.search(
                ltp(compound_indications), window_text["text_between"], flags=re.I
            ),
            re.search(ltp(compound_indications), compound_left_window, flags=re.I),
            re.search(ltp(compound_indications), compound_right_window, flags=re.I),
        ]
    ):
        return POSITIVE

    return ABSTAIN


@labeling_function()
def LF_CtD_TRIAL(c):

    if re.search(ltp(trial_indications), c.text, flags=re.I):
        return POSITIVE

    return ABSTAIN


@labeling_function()
def LF_CD_IN_SERIES(c):
    """
    This label function is designed to look for a mention being caught
    in a series of other genes or compounds
    """
    if len(re.findall(r",", c.text)) >= 2:
        if re.search(", and", c.text):
            return NEGATIVE
    if re.search(r"\(a\)|\(b\)|\(c\)", c.text):
        return NEGATIVE
    return ABSTAIN


method_indication = {
    "investigated (the effect of|in)",
    "was assessed by",
    "assessed",
    "compared with",
    "compared to",
    "were analyzed",
    "evaluated in",
    "examination of",
    "examined in",
    "quantified in" "quantification by",
    "we review",
    "was measured",
    "we(re)? studied",
    "we measured",
    "derived from",
    "regulation of",
    "(are|is) discussed",
    "to measure",
    "to study",
    "to explore",
    "detection of",
    "authors summarize",
    "responsiveness of",
    "used alone",
    "blunting of",
    "measurement of",
    "detection of",
    "occurence of",
    "our objective was",
    "to test the hypothesis",
    "studied in",
    "were reviewed",
    "randomized study",
    "this report considers",
    "was administered",
    "determinations of",
    "we examine",
    "we evaluated",
    "to establish",
    "were selected",
    "authors determmined",
    "we investigated",
    "to assess",
    "analyses were done",
    "useful tool for the study of",
    r"^The effect of",
    "were investigated",
    "to evaluate",
    "study was conducted",
    "to assess",
    "authors applied",
    "were determined",
}

title_indication = {
    "impact of",
    "effects of",
    "the use of",
    "understanding {{B}}( preconditioning)?",
}


@labeling_function()
def LF_CD_METHOD_DESC(c):
    """
    This label function is designed to look for phrases
    that imply a sentence is description an experimental design
    """
    if re.search(ltp(method_indication), c.text, flags=re.I):
        return NEGATIVE

    return ABSTAIN


@labeling_function()
def LF_CD_TITLE(c):
    """
    This label function is designed to look for phrases
    that imply a sentence is the title
    """

    if any(
        [
            re.search(r"^(\[|\[ )?" + ltp(title_indication), c.text, flags=re.I),
            re.search(ltp(title_indication) + r"$", c.text, flags=re.I),
            "(author's transl)" in c.text,
            ":" in c.text,
        ]
    ):
        return NEGATIVE

    return ABSTAIN


@labeling_function()
def LF_CtD_NO_CONCLUSION(c):
    """
    This label function fires a -1 if the number of negative label functinos is greater than the number
    of positive label functions.
    The main idea behind this label function is add support to sentences that could
    mention a possible disease gene association.
    """
    positive_num = np.sum(
        [
            LF_CtD_TREATS(c) == POSITIVE,
            LF_CtD_WEAKLY_TREATS(c) == POSITIVE,
            LF_CtD_COMPOUND_INDICATION(c) == POSITIVE,
            LF_CtD_TRIAL(c) == POSITIVE,
            LF_CD_CHECK_DEPRESSION_USAGE(c) == NEGATIVE,
        ]
    )
    negative_num = np.sum(
        [LF_CD_METHOD_DESC(c) == NEGATIVE, LF_CD_IN_SERIES(c) == NEGATIVE]
    )
    if positive_num - negative_num >= 1:
        return ABSTAIN

    return NEGATIVE


@labeling_function()
def LF_CtD_CONCLUSION(c):
    """
    This label function fires a 1 if the number of positive label functions is greater than the number
    of negative label functions.
    The main idea behind this label function is add support to sentences that could
    mention a possible disease gene association
    """
    if not LF_CtD_NO_CONCLUSION(c) == NEGATIVE:
        return POSITIVE

    return ABSTAIN


@labeling_function()
def LF_CD_DISTANCE_SHORT(c):
    """
    This LF is designed to make sure that the compound mention
    and the gene mention aren't right next to each other.
    """
    window_text = get_text_in_windows(c)
    return NEGATIVE if len(window_text["text_between"].split(" ")) <= 2 else ABSTAIN


@labeling_function()
def LF_CD_DISTANCE_LONG(c):
    """
    This LF is designed to make sure that the compound mention
    and the gene mention aren't too far from each other.
    """
    window_text = get_text_in_windows(c)
    return NEGATIVE if len(window_text["text_between"].split(" ")) > 25 else ABSTAIN


@labeling_function()
def LF_CD_ALLOWED_DISTANCE(c):
    """
    This LF is designed to make sure that the compound mention
    and the gene mention are in an acceptable distance between
    each other
    """
    return (
        ABSTAIN
        if any(
            [LF_CD_DISTANCE_LONG(c) == NEGATIVE, LF_CD_DISTANCE_SHORT(c) == NEGATIVE]
        )
        else POSITIVE
    )


@labeling_function()
def LF_CD_NO_VERB(c):
    """
    This label function is designed to fire if a given
    sentence doesn't contain a verb. Helps cut out some of the titles
    hidden in Pubtator abstracts
    """
    if len(list(filter(lambda x: "VB" in x, c.pos_tag))) != 0:
        return ABSTAIN

    if "correlates with" in c.text:
        return ABSTAIN

    return NEGATIVE


"""
Bi-Clustering LFs
"""
path = (
    pathlib.Path(__file__)
    .joinpath("../../dependency_cluster/chemical_disease_bicluster_results.tsv.xz")
    .resolve()
)
cd_bicluster_dep_df = pd.read_csv(path, sep="\t")
cd_cat_codes = ["T", "C", "Sa", "Pr", "Pa", "J", "Mp"]
cd_dep_path_mapper = create_dep_mapper(cd_bicluster_dep_df, cd_cat_codes)


@labeling_function()
def LF_CD_BICLUSTER_TREATMENT(c):
    """
    This label function uses the bicluster data located in the
    A global network of biomedical relationships
    """
    return match_dep_path(c.dep_path, cd_dep_path_mapper, "T", POSITIVE, ABSTAIN)


@labeling_function()
def LF_CD_BICLUSTER_INHIBITS(c):
    """
    This label function uses the bicluster data located in the
    A global network of biomedical relationships
    """
    return match_dep_path(c.dep_path, cd_dep_path_mapper, "C", POSITIVE, ABSTAIN)


@labeling_function()
def LF_CD_BICLUSTER_SIDE_EFFECT(c):
    """
    This label function uses the bicluster data located in the
    A global network of biomedical relationships
    """
    return match_dep_path(c.dep_path, cd_dep_path_mapper, "Sa", NEGATIVE, ABSTAIN)


@labeling_function()
def LF_CD_BICLUSTER_PREVENTS(c):
    """
    This label function uses the bicluster data located in the
    A global network of biomedical relationships
    """
    return match_dep_path(c.dep_path, cd_dep_path_mapper, "Pr", POSITIVE, ABSTAIN)


@labeling_function()
def LF_CD_BICLUSTER_ALLEVIATES(c):
    """
    This label function uses the bicluster data located in the
    A global network of biomedical relationships
    """
    return match_dep_path(c.dep_path, cd_dep_path_mapper, "Pa", NEGATIVE, ABSTAIN)


@labeling_function()
def LF_CD_BICLUSTER_DISEASE_ROLE(c):
    """
    This label function uses the bicluster data located in the
    A global network of biomedical relationships
    """
    return match_dep_path(c.dep_path, cd_dep_path_mapper, "J", POSITIVE, ABSTAIN)


@labeling_function()
def LF_CD_BICLUSTER_BIOMARKERS(c):
    """
    This label function uses the bicluster data located in the
    A global network of biomedical relationships
    """
    return match_dep_path(c.dep_path, cd_dep_path_mapper, "Mp", POSITIVE, ABSTAIN)


"""
RETRUN LFs to Notebook
"""

LFS = OrderedDict(
    {
        "distant_supervision": {
            "LF_HETNET_PHARMACOTHERAPYDB": LF_HETNET_PHARMACOTHERAPYDB,
            "LF_HETNET_CD_ABSENT": LF_HETNET_CD_ABSENT,
            "LF_CD_CHECK_DISEASE_TAG": LF_CD_CHECK_DISEASE_TAG,
        },
        "text_patterns": {
            "LF_CtD_TREATS": LF_CtD_TREATS,
            "LF_CD_CHECK_DEPRESSION_USAGE": LF_CD_CHECK_DEPRESSION_USAGE,
            "LF_CtD_WEAKLY_TREATS": LF_CtD_WEAKLY_TREATS,
            "LF_CD_INCORRECT_COMPOUND": LF_CD_INCORRECT_COMPOUND,
            "LF_CtD_COMPOUND_INDICATION": LF_CtD_COMPOUND_INDICATION,
            "LF_CtD_TRIAL": LF_CtD_TRIAL,
            "LF_CD_IN_SERIES": LF_CD_IN_SERIES,
            "LF_CD_METHOD_DESC": LF_CD_METHOD_DESC,
            "LF_CD_TITLE": LF_CD_TITLE,
            "LF_CtD_NO_CONCLUSION": LF_CtD_NO_CONCLUSION,
            "LF_CtD_CONCLUSION": LF_CtD_CONCLUSION,
            "LF_CD_DISTANCE_SHORT": LF_CD_DISTANCE_SHORT,
            "LF_CD_DISTANCE_LONG": LF_CD_DISTANCE_LONG,
            "LF_CD_ALLOWED_DISTANCE": LF_CD_ALLOWED_DISTANCE,
            "LF_CD_NO_VERB": LF_CD_NO_VERB,
        },
        "domain_heuristics": {
            "LF_CD_BICLUSTER_TREATMENT": LF_CD_BICLUSTER_TREATMENT,
            "LF_CD_BICLUSTER_INHIBITS": LF_CD_BICLUSTER_INHIBITS,
            "LF_CD_BICLUSTER_SIDE_EFFECT": LF_CD_BICLUSTER_SIDE_EFFECT,
            "LF_CD_BICLUSTER_PREVENTS": LF_CD_BICLUSTER_PREVENTS,
            "LF_CD_BICLUSTER_ALLEVIATES": LF_CD_BICLUSTER_ALLEVIATES,
            "LF_CD_BICLUSTER_DISEASE_ROLE": LF_CD_BICLUSTER_DISEASE_ROLE,
            "LF_CD_BICLUSTER_BIOMARKERS": LF_CD_BICLUSTER_BIOMARKERS,
        },
    }
)
