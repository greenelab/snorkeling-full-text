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
    .joinpath("../../knowledge_bases/gene_interacts_gene.tsv.xz")
    .resolve()
)
print(path)
pair_df = pd.read_csv(path, dtype={"sources": str}, sep="\t")
knowledge_base = set()
for row in pair_df.itertuples():
    if not row.sources or pd.isnull(row.sources):
        continue

    for source in row.sources.split("|"):
        key = str(row.gene1_id), str(row.gene2_id), source.lower()
        knowledge_base.add(key)


# Human Interactome Datasets
@labeling_function()
def LF_HETNET_HI_I_05(c):
    return (
        POSITIVE
        if (c.gene1_cid, c.gene2_cid, "hi-i-05") in knowledge_base
        or (c.gene2_cid, c.gene1_cid, "hi-i-05") in knowledge_base
        else ABSTAIN
    )


@labeling_function()
def LF_HETNET_VENKATESAN_09(c):
    return (
        POSITIVE
        if (c.gene1_cid, c.gene2_cid, "venkatesan-09") in knowledge_base
        or (c.gene2_cid, c.gene1_cid, "venkatesan-09") in knowledge_base
        else ABSTAIN
    )


@labeling_function()
def LF_HETNET_YU_11(c):
    return (
        POSITIVE
        if (c.gene1_cid, c.gene2_cid, "yu-11") in knowledge_base
        or (c.gene2_cid, c.gene1_cid, "yu-11") in knowledge_base
        else ABSTAIN
    )


@labeling_function()
def LF_HETNET_HI_II_14(c):
    return (
        POSITIVE
        if (c.gene1_cid, c.gene2_cid, "hi-ii-14") in knowledge_base
        or (c.gene2_cid, c.gene1_cid, "hi-ii-14") in knowledge_base
        else ABSTAIN
    )


@labeling_function()
def LF_HETNET_LIT_BM_13(c):
    return (
        POSITIVE
        if (c.gene1_cid, c.gene2_cid, "lit-bm-13") in knowledge_base
        or (c.gene2_cid, c.gene1_cid, "lit-bm-13") in knowledge_base
        else ABSTAIN
    )


# Incomplete Interactome
@labeling_function()
def LF_HETNET_II_BINARY(c):
    return (
        POSITIVE
        if (c.gene1_cid, c.gene2_cid, "ii-binary") in knowledge_base
        or (c.gene2_cid, c.gene1_cid, "ii-binary") in knowledge_base
        else ABSTAIN
    )


@labeling_function()
def LF_HETNET_II_LITERATURE(c):
    return (
        POSITIVE
        if (c.gene1_cid, c.gene2_cid, "ii-literature") in knowledge_base
        or (c.gene2_cid, c.gene1_cid, "ii-literature") in knowledge_base
        else ABSTAIN
    )


# Hetionet
@labeling_function()
def LF_HETNET_HETIO_DAG(c):
    return (
        POSITIVE
        if (c.gene1_cid, c.gene2_cid, "hetio-dag") in knowledge_base
        or (c.gene2_cid, c.gene1_cid, "hetio-dag") in knowledge_base
        else ABSTAIN
    )


@labeling_function()
def LF_HETNET_GiG_ABSENT(c):
    return (
        ABSTAIN
        if any(
            [
                LF_HETNET_HI_I_05(c) == POSITIVE,
                LF_HETNET_VENKATESAN_09(c) == POSITIVE,
                LF_HETNET_YU_11(c) == POSITIVE,
                LF_HETNET_HI_II_14(c) == POSITIVE,
                LF_HETNET_LIT_BM_13(c) == POSITIVE,
                LF_HETNET_II_BINARY(c) == POSITIVE,
                LF_HETNET_II_LITERATURE(c) == POSITIVE,
                LF_HETNET_HETIO_DAG(c) == POSITIVE,
            ]
        )
        else NEGATIVE
    )


"""
2. SENTENCE PATTERN MATCHING
"""
binding_identifiers = {
    "interact(s with|ed)",
    "bind(s|ing)?",
    "phosphorylat(es|ion)",
    "heterodimeriz(e|ation)",
    "component binding",
    "multiple ligand binding",
    "cross-link(ing|ed)?",
    "mediates",
    "potential target for",
    "interaction( of|s with)",
    "receptor binding",
    "reactions",
    "phosphorylation by",
    "up-regulated by",
    "coactivators",
    "bound to",
}

cell_indications = {
    "cell(s)?",
    r"\+",
    "-",
    "immunophenotyping",
    "surface marker analysis",
}

compound_indications = {"inhibitors", "therapy"}

upregulates_identifiers = {
    "elevated( serum)?",
    "amplification of",
    "enhance(s|d)",
    "phsophorylation",
    "transcriptional activation",
    "potentiated",
    "stimulate production",
    "up-regulated",
}

downregulates_identifiers = {
    "decreased",
    "depressed",
    "inhibitory action",
    "competitive inhibition",
    "defective",
    "inihibit(ed|s)",
    "abrogated",
}

regulation_identifiers = {"mediates", "modulates", "stimulate production"}

association_identifiers = {"associate(s|d)( with)?", "statsitically significant"}

bound_identifiers = {
    "heterodimer(s)?",
    "receptor(s)?",
    "enzyme",
    "binding protein",
    "mediator",
}

gene_identifiers = {"variant(s)?", "markers", "gene", "antigen", "mutations( in| of)"}

gene_adjective = {"responsive", "mediated"}

diagnosis_indication = {"diagnostic markers", "diagnosis of"}

method_indication = {
    "was determined",
    "was assayed",
    "removal of",
    "to assess",
    "the effect of",
    "was studied",
    "coeluted with",
    "we evaluated",
}


@labeling_function()
def LF_GiG_BINDING_IDENTIFICATIONS(c):
    window_text = get_text_in_windows(c)
    gene1_left_window = " ".join(window_text["left_window"][0])
    gene1_right_window = " ".join(window_text["right_window"][0])
    gene2_left_window = " ".join(window_text["left_window"][1])
    gene2_right_window = " ".join(window_text["right_window"][1])

    if any(
        [
            re.search(
                ltp(binding_identifiers), window_text["text_between"], flags=re.I
            ),
            re.search(
                ltp(binding_identifiers),
                gene1_left_window + " " + gene1_right_window,
                flags=re.I,
            ),
            re.search(
                ltp(binding_identifiers),
                gene2_left_window + " " + gene2_right_window,
                flags=re.I,
            ),
        ]
    ):
        return POSITIVE

    return ABSTAIN


@labeling_function()
def LF_GiG_CELL_IDENTIFICATIONS(c):
    window_text = get_text_in_windows(c)
    gene1_left_window = " ".join(window_text["left_window"][0])
    gene1_right_window = " ".join(window_text["right_window"][0])
    gene2_left_window = " ".join(window_text["left_window"][1])
    gene2_right_window = " ".join(window_text["right_window"][1])

    if any(
        [
            re.search(
                ltp(cell_indications),
                gene1_left_window + " " + gene1_right_window,
                flags=re.I,
            ),
            re.search(
                ltp(cell_indications),
                gene2_left_window + " " + gene2_right_window,
                flags=re.I,
            ),
        ]
    ):
        return NEGATIVE

    return ABSTAIN


@labeling_function()
def LF_GiG_COMPOUND_IDENTIFICATIONS(c):
    window_text = get_text_in_windows(c)
    gene1_right_window = " ".join(window_text["right_window"][0])
    gene2_right_window = " ".join(window_text["right_window"][1])

    if any(
        [
            re.search(
                ltp(compound_indications), window_text["text_between"], flags=re.I
            ),
            re.search(ltp(compound_indications), gene1_right_window, flags=re.I),
            re.search(ltp(compound_indications), gene2_right_window, flags=re.I),
        ]
    ):
        return NEGATIVE

    return ABSTAIN


@labeling_function()
def LF_GiG_UPREGULATES(c):
    window_text = get_text_in_windows(c)
    gene1_right_window = " ".join(window_text["right_window"][0])
    gene2_right_window = " ".join(window_text["right_window"][1])

    if any(
        [
            re.search(
                ltp(upregulates_identifiers), window_text["text_between"], flags=re.I
            ),
            re.search(ltp(upregulates_identifiers), gene1_right_window, flags=re.I),
            re.search(ltp(upregulates_identifiers), gene2_right_window, flags=re.I),
        ]
    ):
        return NEGATIVE

    return ABSTAIN


@labeling_function()
def LF_GiG_DOWNREGULATES(c):
    window_text = get_text_in_windows(c)
    gene1_right_window = " ".join(window_text["right_window"][0])
    gene2_right_window = " ".join(window_text["right_window"][1])

    if any(
        [
            re.search(
                ltp(downregulates_identifiers), window_text["text_between"], flags=re.I
            ),
            re.search(ltp(downregulates_identifiers), gene1_right_window, flags=re.I),
            re.search(ltp(downregulates_identifiers), gene2_right_window, flags=re.I),
        ]
    ):
        return NEGATIVE

    return ABSTAIN


@labeling_function()
def LF_GiG_REGULATION(c):
    window_text = get_text_in_windows(c)
    if any(
        [
            LF_GiG_UPREGULATES(c) == NEGATIVE,
            LF_GiG_DOWNREGULATES(c) == NEGATIVE,
            re.search(
                ltp(regulation_identifiers), window_text["text_between"], flags=re.I
            ),
        ]
    ):
        return NEGATIVE

    return ABSTAIN


@labeling_function()
def LF_GiG_ASSOCIATION(c):
    window_text = get_text_in_windows(c)
    gene1_right_window = " ".join(window_text["right_window"][0])
    gene2_right_window = " ".join(window_text["right_window"][1])

    if any(
        [
            re.search(
                ltp(association_identifiers), window_text["text_between"], flags=re.I
            ),
            re.search(ltp(association_identifiers), gene1_right_window, flags=re.I),
            re.search(ltp(association_identifiers), gene2_right_window, flags=re.I),
        ]
    ):
        return NEGATIVE

    return ABSTAIN


@labeling_function()
def LF_GiG_BOUND_IDENTIFIERS(c):
    window_text = get_text_in_windows(c)
    gene1_left_window = " ".join(window_text["left_window"][0])
    gene1_right_window = " ".join(window_text["right_window"][0])
    gene2_left_window = " ".join(window_text["left_window"][1])
    gene2_right_window = " ".join(window_text["right_window"][1])

    if any(
        [
            re.search(
                ltp(bound_identifiers),
                gene1_left_window + " " + gene1_right_window,
                flags=re.I,
            ),
            re.search(
                ltp(bound_identifiers),
                gene2_left_window + " " + gene2_right_window,
                flags=re.I,
            ),
        ]
    ):
        return POSITIVE

    return ABSTAIN


@labeling_function()
def LF_GiG_GENE_IDENTIFIERS(c):
    window_text = get_text_in_windows(c)
    gene1_left_window = " ".join(window_text["left_window"][0])
    gene1_right_window = " ".join(window_text["right_window"][0])
    gene2_left_window = " ".join(window_text["left_window"][1])
    gene2_right_window = " ".join(window_text["right_window"][1])

    if any(
        [
            re.search(
                ltp(gene_identifiers),
                gene1_left_window + " " + gene1_right_window,
                flags=re.I,
            ),
            re.search(
                ltp(gene_identifiers),
                gene2_left_window + " " + gene2_right_window,
                flags=re.I,
            ),
        ]
    ):
        return POSITIVE

    return ABSTAIN


@labeling_function()
def LF_GiG_GENE_ADJECTIVE(c):
    window_text = get_text_in_windows(c)
    entity_one_start = window_text["entity_columns"]["entity_one_start"]
    entity_one_end = window_text["entity_columns"]["entity_one_end"]
    entity_two_start = window_text["entity_columns"]["entity_two_start"]
    entity_two_end = window_text["entity_columns"]["entity_two_end"]

    if any(
        [
            # "-" in c.gene1_span,
            re.search(
                ltp(gene_adjective),
                " ".join(c.word[c[entity_one_start] : c[entity_one_end]]),
                flags=re.I,
            ),
            # "-" in c.gene2_span,
            re.search(
                ltp(gene_adjective),
                " ".join(c.word[c[entity_two_start] : c[entity_two_end]]),
                flags=re.I,
            ),
        ]
    ):
        return NEGATIVE

    return ABSTAIN


@labeling_function()
def LF_GiG_DIAGNOSIS_IDENTIFIERS(c):
    window_text = get_text_in_windows(c)
    if re.search(ltp(diagnosis_indication), window_text["text_between"], flags=re.I):
        return NEGATIVE

    return ABSTAIN


@labeling_function()
def LF_GiG_METHOD_DESC(c):
    window_text = get_text_in_windows(c)
    if any(
        [
            re.search(ltp(method_indication), c.text, flags=re.I),
            re.search(ltp(method_indication), window_text["text_between"], flags=re.I),
        ]
    ):
        return NEGATIVE

    return ABSTAIN


@labeling_function()
def LF_GiG_PARENTHESIS(c):
    window_text = get_text_in_windows(c)
    entity_one_start = window_text["entity_columns"]["entity_one_start"]
    entity_one_end = window_text["entity_columns"]["entity_one_end"]
    entity_two_start = window_text["entity_columns"]["entity_two_start"]
    entity_two_end = window_text["entity_columns"]["entity_two_end"]

    if any(
        [
            (
                ")" in " ".join(c.word[c[entity_one_start] : c[entity_one_end]])
                and LF_GG_DISTANCE_SHORT(c)
            ),
            (
                ")" in " ".join(c.word[c[entity_two_start] : c[entity_two_end]])
                and LF_GG_DISTANCE_SHORT(c)
            ),
        ]
    ):
        return NEGATIVE

    return ABSTAIN


@labeling_function()
def LF_GG_IN_SERIES(c):
    if len(re.findall(r",", c.text)) >= 2:
        if re.search(", and", c.text):
            return NEGATIVE
    return ABSTAIN


@labeling_function()
def LF_GiG_NO_CONCLUSION(c):
    positive_num = np.sum(
        [
            LF_GiG_BINDING_IDENTIFICATIONS(c) == POSITIVE,
            LF_GiG_GENE_IDENTIFIERS(c) == POSITIVE,
            LF_GiG_BOUND_IDENTIFIERS(c) == POSITIVE,
            LF_GiG_UPREGULATES(c) == NEGATIVE,
            LF_GiG_DOWNREGULATES(c) == NEGATIVE,
        ]
    )
    negative_num = np.abs(
        np.sum(
            [
                LF_GiG_CELL_IDENTIFICATIONS(c) == NEGATIVE,
                LF_GiG_COMPOUND_IDENTIFICATIONS(c) == NEGATIVE,
                LF_GG_NO_VERB(c) == NEGATIVE,
                LF_GiG_PARENTHESIS(c) == NEGATIVE,
                LF_GiG_DIAGNOSIS_IDENTIFIERS(c) == NEGATIVE,
            ]
        )
    )
    if positive_num - negative_num >= 1:
        return ABSTAIN

    return NEGATIVE


@labeling_function()
def LF_GiG_CONCLUSION(c):
    if not LF_GiG_NO_CONCLUSION(c) == NEGATIVE:

        if LF_GiG_UPREGULATES(c) == NEGATIVE or LF_GiG_DOWNREGULATES(c) == NEGATIVE:
            return NEGATIVE

        return POSITIVE

    return ABSTAIN


@labeling_function()
def LF_GG_DISTANCE_SHORT(c):
    """
    This LF is designed to make sure that the disease mention
    and the gene mention aren't right next to each other.
    """
    window_text = get_text_in_windows(c)
    return NEGATIVE if len(window_text["text_between"].split(" ")) <= 1 else ABSTAIN


@labeling_function()
def LF_GG_DISTANCE_LONG(c):
    """
    This LF is designed to make sure that the disease mention
    and the gene mention aren't too far from each other.
    """
    window_text = get_text_in_windows(c)
    return NEGATIVE if len(window_text["text_between"].split(" ")) > 25 else ABSTAIN


@labeling_function()
def LF_GG_ALLOWED_DISTANCE(c):
    """
    This LF is designed to make sure that the disease mention
    and the gene mention are in an acceptable distance between
    each other
    """
    return (
        ABSTAIN
        if any(
            [LF_GG_DISTANCE_LONG(c) == NEGATIVE, LF_GG_DISTANCE_SHORT(c) == NEGATIVE]
        )
        else POSITIVE
    )


@labeling_function()
def LF_GG_NO_VERB(c):
    if len(list(filter(lambda x: "VB" in x, c.pos_tag))) != 0:
        return ABSTAIN

    return NEGATIVE


"""
3. Domain Heuristics
"""
path = (
    pathlib.Path(__file__)
    .joinpath("../../dependency_cluster/gene_gene_bicluster_results.tsv.xz")
    .resolve()
)
gg_bicluster_dep_df = pd.read_csv(path, sep="\t")
gg_cat_codes = ["B", "W", "V+", "E+", "E", "I", "H", "Rg", "Q"]
gg_dep_path_mapper = create_dep_mapper(gg_bicluster_dep_df, gg_cat_codes)


@labeling_function()
def LF_GG_BICLUSTER_BINDING(c):
    return match_dep_path(c.dep_path, gg_dep_path_mapper, "B", POSITIVE, ABSTAIN)


@labeling_function()
def LF_GG_BICLUSTER_ENHANCES(c):
    return match_dep_path(c.dep_path, gg_dep_path_mapper, "W", POSITIVE, ABSTAIN)


@labeling_function()
def LF_GG_BICLUSTER_ACTIVATES(c):
    return match_dep_path(c.dep_path, gg_dep_path_mapper, "V+", POSITIVE, ABSTAIN)


@labeling_function()
def LF_GG_BICLUSTER_INCREASES_EXPRESSION(c):
    return match_dep_path(c.dep_path, gg_dep_path_mapper, "E+", NEGATIVE, ABSTAIN)


@labeling_function()
def LF_GG_BICLUSTER_AFFECTS_EXPRESSION(c):
    return match_dep_path(c.dep_path, gg_dep_path_mapper, "E", POSITIVE, ABSTAIN)


@labeling_function()
def LF_GG_BICLUSTER_SIGNALING(c):
    return match_dep_path(c.dep_path, gg_dep_path_mapper, "I", POSITIVE, ABSTAIN)


@labeling_function()
def LF_GG_BICLUSTER_IDENTICAL_PROTEIN(c):
    return match_dep_path(c.dep_path, gg_dep_path_mapper, "H", NEGATIVE, ABSTAIN)


@labeling_function()
def LF_GG_BICLUSTER_REGULATION(c):
    return match_dep_path(c.dep_path, gg_dep_path_mapper, "Rg", POSITIVE, ABSTAIN)


@labeling_function()
def LF_GG_BICLUSTER_CELL_PRODUCTION(c):
    return match_dep_path(c.dep_path, gg_dep_path_mapper, "Q", NEGATIVE, ABSTAIN)


LFS = OrderedDict(
    {
        "distant_supervision": {
            "LF_HETNET_HI_I_05": LF_HETNET_HI_I_05,
            "LF_HETNET_VENKATESAN_09": LF_HETNET_VENKATESAN_09,
            "LF_HETNET_YU_11": LF_HETNET_YU_11,
            "LF_HETNET_HI_II_14": LF_HETNET_HI_II_14,
            "LF_HETNET_LIT_BM_13": LF_HETNET_LIT_BM_13,
            "LF_HETNET_II_BINARY": LF_HETNET_II_BINARY,
            "LF_HETNET_II_LITERATURE": LF_HETNET_II_LITERATURE,
            "LF_HETNET_HETIO_DAG": LF_HETNET_HETIO_DAG,
            "LF_HETNET_GiG_ABSENT": LF_HETNET_GiG_ABSENT,
        },
        "text_patterns": {
            "LF_GiG_BINDING_IDENTIFICATIONS": LF_GiG_BINDING_IDENTIFICATIONS,
            "LF_GiG_CELL_IDENTIFICATIONS": LF_GiG_CELL_IDENTIFICATIONS,
            "LF_GiG_COMPOUND_IDENTIFICATIONS": LF_GiG_COMPOUND_IDENTIFICATIONS,
            "LF_GiG_UPREGULATES": LF_GiG_UPREGULATES,
            "LF_GiG_DOWNREGULATES": LF_GiG_DOWNREGULATES,
            "LF_GiG_REGULATION": LF_GiG_REGULATION,
            "LF_GiG_ASSOCIATION": LF_GiG_ASSOCIATION,
            "LF_GiG_BOUND_IDENTIFIERS": LF_GiG_BOUND_IDENTIFIERS,
            "LF_GiG_GENE_IDENTIFIERS": LF_GiG_GENE_IDENTIFIERS,
            "LF_GiG_GENE_ADJECTIVE": LF_GiG_GENE_ADJECTIVE,
            "LF_GiG_DIAGNOSIS_IDENTIFIERS": LF_GiG_DIAGNOSIS_IDENTIFIERS,
            "LF_GiG_METHOD_DESC": LF_GiG_METHOD_DESC,
            "LF_GiG_PARENTHESIS": LF_GiG_PARENTHESIS,
            "LF_GG_IN_SERIES": LF_GG_IN_SERIES,
            "LF_GiG_NO_CONCLUSION": LF_GiG_NO_CONCLUSION,
            "LF_GiG_CONCLUSION": LF_GiG_CONCLUSION,
            "LF_GG_DISTANCE_SHORT": LF_GG_DISTANCE_SHORT,
            "LF_GG_DISTANCE_LONG": LF_GG_DISTANCE_LONG,
            "LF_GG_ALLOWED_DISTANCE": LF_GG_ALLOWED_DISTANCE,
            "LF_GG_NO_VERB": LF_GG_NO_VERB,
        },
        "domain_heuristics": {
            "LF_GG_BICLUSTER_BINDING": LF_GG_BICLUSTER_BINDING,
            "LF_GG_BICLUSTER_ENHANCES": LF_GG_BICLUSTER_ENHANCES,
            "LF_GG_BICLUSTER_ACTIVATES": LF_GG_BICLUSTER_ACTIVATES,
            "LF_GG_BICLUSTER_AFFECTS_EXPRESSION": LF_GG_BICLUSTER_AFFECTS_EXPRESSION,
            "LF_GG_BICLUSTER_INCREASES_EXPRESSION": LF_GG_BICLUSTER_INCREASES_EXPRESSION,
            "LF_GG_BICLUSTER_SIGNALING": LF_GG_BICLUSTER_SIGNALING,
            "LF_GG_BICLUSTER_IDENTICAL_PROTEIN": LF_GG_BICLUSTER_IDENTICAL_PROTEIN,
            "LF_GG_BICLUSTER_CELL_PRODUCTION": LF_GG_BICLUSTER_CELL_PRODUCTION,
        },
    }
)
