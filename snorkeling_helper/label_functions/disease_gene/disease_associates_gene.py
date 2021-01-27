# Global Imports for Positive (1) Negative (0) and ABSTAIN (-1)
from ..global_helpers import (
    ABSTAIN,
    NEGATIVE,
    POSITIVE,
    OrderedDict,
    pathlib,
    re,
    pd,
    ltp,
    labeling_function,
    get_tagged_text,
    get_tokens_between,
    get_token_windows
)
import numpy as np

"""
DISTANT SUPERVISION
"""
path = (
    pathlib.Path(__file__)
    .joinpath("../../knowledge_bases/disease_associates_gene.tsv.xz")
    .resolve()
)
pair_df = pd.read_csv(path, dtype={"sources": str}, sep="\t")
knowledge_base = set()
for row in pair_df.itertuples():
    if not row.sources or pd.isnull(row.sources):
        continue
    for source in row.sources.split("|"):
        key = str(row.entrez_gene_id), row.doid_id, source
        knowledge_base.add(key)


@labeling_function()
def LF_HETNET_DISEASES(c):
    """
    This label function returns 1 if the given Disease Gene pair is
    located in the Diseases database
    """
    return (
        POSITIVE
        if (c.gene_cid, c.disease_cid, "DISEASES") in knowledge_base
        else ABSTAIN
    )


@labeling_function()
def LF_HETNET_DOAF(c):
    """
    This label function returns 1 if the given Disease Gene pair is
    located in the DOAF database
    """
    return (
        POSITIVE if (c.gene_cid, c.disease_cid, "DOAF") in knowledge_base else ABSTAIN
    )


@labeling_function()
def LF_HETNET_DisGeNET(c):
    """
    This label function returns 1 if the given Disease Gene pair is
    located in the DisGeNET database
    """
    return (
        POSITIVE
        if (c.gene_cid, c.disease_cid, "DisGeNET") in knowledge_base
        else ABSTAIN
    )


@labeling_function()
def LF_HETNET_GWAS(c):
    """
    This label function returns 1 if the given Disease Gene pair is
    located in the GWAS database
    """
    return (
        POSITIVE
        if (c.gene_cid, c.disease_cid, "GWAS Catalog") in knowledge_base
        else ABSTAIN
    )


@labeling_function()
def LF_HETNET_DaG_ABSENT(c):
    """
    This label function fires -1 if the given Disease Gene pair does not appear
    in the databases above.
    """
    return (
        ABSTAIN
        if any(
            [
                LF_HETNET_DISEASES(c) == POSITIVE,
                LF_HETNET_DOAF(c) == POSITIVE,
                LF_HETNET_DisGeNET(c) == POSITIVE,
                LF_HETNET_GWAS(c) == POSITIVE,
            ]
        )
        else NEGATIVE
    )


"""
SENTENCE PATTERN MATCHING
"""

biomarker_indicators = {
    "useful marker of",
    "useful in predicting",
    "modulates the expression of",
    "expressed in",
    "prognostic marker",
    "tissue marker",
    "tumor marker",
    "level(s)? (of|in)",
    "high concentrations of",
    "(cytoplamsic )?concentration of",
    "have fewer",
    "quantification of",
    "evaluation of",
    "hypersecreted by",
    "assess the presensece of",
    "stained postively for",
    "overproduced",
    "prognostic factor",
    "characterized by a marked",
    "plasma levels of",
    "had elevated",
    "were detected",
    "exaggerated response to",
    "serum",
    "expressed on",
    "overexpression of",
    "plasma",
    "over-expression",
    "high expression" "detection marker",
    "increased",
    "was enhanced",
    "was elevated in",
    "expression (in|of)",
    "significantly higher (concentrations of|in)",
    "higher and lower amounts of",
    "measurement of",
    "levels discriminate",
    "potential biomarker of",
    "elevated serum levels",
    "elevated",
}

cellular_activity = {
    "positive immunostaining",
    "stronger immunoreactivity",
    "in vitro kinase activity",
    "incude proliferation",
    "apoptosis in",
    "early activation",
    "activation in",
    "depletion inhibited",
    "transcriptional activity",
    "transcriptionally activates",
    "anti-tumor cell efficacy",
    "suppresses the development and progression",
    "secret an adaquate amount of",
    "epigenetic alteration of",
    "actively transcribe",
    "decreased {{B}} production in",
    "rna targeting {{B}}",
    "suppresses growth of human",
    "inhibits growth of",
    "partial agonist",
    "mediates {{B}} pi3k signaling",
    "induces apoptosis",
    "antitumor activity of",
    "{{B}} stained",
    "(?<!not ){{B}} agonist(s)?",
    "produc(e|tion).*{{B}}",
    "sensitizes {{A}}",
    "endocannabinoid involvement",
    "epigenetically regulates",
    "actively transcribe",
    "re-expression of {{B}}",
}

direct_association = {
    "association (with|of)",
    "association between",
    "associated with",
    "associated between",
    "stimulated by",
    "correlat(ed|es|ion)? between",
    "correlat(e|ed|es|ion)? with",
    "significant ultradian variation",
    "showed (that|loss)",
    "found in",
    "involved in",
    "central role in",
    "inhibited by",
    "greater for",
    "indicative of",
    "increased production of",
    "control the extent of",
    "secreted by",
    "detected in",
    "positive for",
    "to be mediated",
    "was produced by",
    "stimulates",
    "precipitated by",
    "affects",
    "counteract cholinergic deficits",
    "mediator of",
    "candidate gene",
    "categorized",
    "positive correlation",
    "regulated by",
    "important role in",
    "significant amounts of",
    "to contain",
    "increased risk of",
    "express",
    "susceptibility gene(s)? for",
    "risk factor for",
    "necessary and sufficient to",
    "associated gene",
    "plays crucial role in",
    "common cause of",
    "discriminate",
    "were observed",
}

upregulates = {
    r"\bhigh\b",
    "elevate(d|s)?",
    "greated for",
    "greater in",
    "higher",
    "prevent their degeneration",
    "gain",
    "increased",
    "positive",
    "strong",
    "elevated",
    "upregulated",
    "up-regulat(ed|ion)",
    "higher",
    "was enhanced",
    "over-expression",
    "overexpression",
    "phosphorylates",
    "activated by",
    "significantly higher concentrations of",
    "highly expressed in",
    "3-fold higher expression of",
}

downregulates = {
    r"\blow\b",
    "reduce(d|s)?",
    "(significant(ly)?)? decrease(d|s)?",
    "inhibited by",
    "not higher",
    "unresponsive",
    "under-expression",
    "underexpresed",
    "down-regulat(ed|ion)",
    "downregulated",
    "knockdown",
    "suppressed",
    "negative",
    "weak",
    "lower",
    "suppresses",
    "deletion of",
    "through decrease in",
}

disease_sample_indicators = {
    "tissue",
    "cell",
    "patient",
    "tumor",
    "cancer",
    "carcinoma",
    "cell line",
    "cell-line",
    "group",
    "blood",
    "sera",
    "serum",
    "fluid",
    "subset",
    "case",
}

diagnosis_indicators = {
    "prognostic significance of",
    "prognostic indicator for",
    "prognostic cyosolic factor",
    "prognostic parameter for",
    "prognostic information for",
    "predict(or|ive) of",
    "predictor of prognosis in",
    "indicative of",
    "diagnosis of",
    "was positive for",
    "detection of",
    "determined by",
    "diagnositic sensitivity",
    "dianostic specificity",
    "prognostic factor",
    "variable for the identification",
    "potential therapeutic agent",
    "prognostic parameter for",
    "identification of",
    "psychophysiological index of suicdal risk",
    "reflects clinical activity",
}

no_direct_association = {
    "not significant",
    "not significantly",
    "no association",
    "not associated",
    "no correlation between" "no correlation in",
    "no correlation with",
    "not correlated with",
    "not detected in",
    "not been observed",
    "not appear to be related to",
    "neither",
    "provide evidence against",
    "not a constant",
    "not predictive",
    "nor were they correlated with",
    "lack of",
    "correlation was lost in",
    "no obvious association",
    ", whereas",
    "do not support",
    "not find an association",
    "little is known",
    "does( n't|n't) appear to affect",
    "no way to differentiate",
    "not predictor of",
    "roles are unknown",
    "independent of",
    "no expression of",
    "abscence of",
    "are unknown",
    "not increased in",
    "not been elucidated",
}

weak_association = {
    "not necessarily indicate",
    "the possibility",
    "low correlation",
    "may be.*important",
    "might facillitate",
    "might be closely related to",
    "has potential",
    "maybe a target for",
    "potential (bio)?marker for",
    "implicated in",
    "clinical activity in",
    "may represent",
    "mainly responsible for",
    "we hypothesized",
    "potential contributors",
    "suggests the diagnosis of",
    "suspected of contributing",
}

method_indication = {
    "investigate(d)? (the effect of|in)?",
    "was assessed by",
    "assessed",
    "compared to",
    "w(as|e|ere)? analy(z|s)ed",
    "evaluated in",
    "examination of",
    "examined in",
    "quantified in" "quantification by",
    "we review",
    "(were|was) measured",
    "we(re)?( have)? studied",
    "we measured",
    "derived from",
    "(are|is) discussed",
    "to measure",
    "(prospective|to) study",
    "to explore",
    "detection of",
    "authors summarize",
    "responsiveness of",
    "used alone",
    "blunting of",
    "measurement of",
    "detection of",
    "occurence of",
    "our objective( was)?",
    "to test the hypothesis",
    "studied in",
    "were reviewed",
    "randomized study",
    "this report considers",
    "was administered",
    "determinations of",
    "we examine(d)?",
    "(was|we|were|to) evaluate(d)?",
    "to establish",
    "were selected",
    "(authors|were|we) determined",
    "we investigated",
    "to assess",
    "analyses were done",
    "for the study of",
    r"^The effect of",
    "OBJECTIVE :",
    "PURPOSE :",
    "METHODS :",
    "were applied",
    "EXPERIMENTAL DESIGN :",
    "we explored",
    "the purpose of",
    "to understand how",
    "to examine",
    "was conducted",
    "to determine",
    "we validated",
    "we characterized",
    "aim of (our|this|the) (study|meta-analysis)",
    "developing a",
    "we tested for",
    " was demonstrate(d)?",
    "we describe",
    "were compared",
    "were categorized",
    "was studied",
    "we calculate(d)?",
    "sought to investigate",
    "this study aimed",
    "a study was made",
    "study sought",
}

title_indication = {
    "Effect of",
    "Evaluation of",
    "Clincal value of",
    "Extraction of",
    "Responsiveness of",
    "The potential for",
    "as defined by immunohistochemistry",
    "Comparison between",
    "Characterization of",
    "A case of",
    "Occurrence of",
    "Inborn",
    "Episodic",
    "Detection of",
    "Immunostaining of",
    "Mutational analysis of",
    "Identification of",
    "souble expression of",
    "expression of",
    "genetic determinants of",
    "prolactin levels in",
    "a study on",
    "analysis of",
}
genetic_abnormalities = {
    "deletions (in|of)",
    "mutation(s)? in",
    "polymorphism(s)?",
    "promoter variant(s)?",
    "recombinant human",
    "novel {{B}} gene mutation",
    "pleotropic effects on",
}

context_change_keywords = {
    ", but",
    ", whereas",
    "; however,",
}


def get_text_in_windows(c):
    between_phrases = " ".join(
        get_tokens_between(
            word_array=c.word,
            entity_one_start=c.disease_start,
            entity_one_end=c.disease_end,
            entity_two_start=c.gene_start,
            entity_two_end=c.gene_end,
        )
    )

    gene_left_window, gene_right_window = get_token_windows(
        word_array=c.word,
        entity_offset_start=c.gene_start,
        entity_offset_end=c.gene_end,
        window_size=10,
    )

    disease_left_window, disease_right_window = get_token_windows(
        word_array=c.word,
        entity_offset_start=c.disease_start,
        entity_offset_end=c.disease_end,
        window_size=10,
    )

    return {
        "text_between": between_phrases,
        "left_window": (disease_left_window, gene_left_window),
        "right_window": (disease_right_window, gene_right_window),
    }


@labeling_function()
def LF_DG_IS_BIOMARKER(c):
    """
    This label function examines a sentences to determine of a sentence
    is talking about a biomarker. (A biomarker leads towards D-G assocation
    c - The candidate obejct being passed in
    """
    window_text = get_text_in_windows(c)
    gene_left_window = " ".join(window_text["left_window"][1])
    gene_right_window = " ".join(window_text["right_window"][1])

    # Look ten words to the left or right
    if any(
        [
            re.search(ltp(biomarker_indicators), gene_left_window, flags=re.I),
            re.search(ltp(biomarker_indicators), gene_right_window, flags=re.I),
        ]
    ):
        return POSITIVE

    else:
        return ABSTAIN


@labeling_function()
def LF_DaG_ASSOCIATION(c):
    """
    This LF is designed to test if there is a key phrase that suggests
    a d-g pair is an association.
    """
    window_text = get_text_in_windows(c)
    left_window = " ".join(
        window_text["left_window"][0] + window_text["left_window"][1]
    )
    right_window = " ".join(
        window_text["right_window"][0] + window_text["right_window"][1]
    )

    if any(
        [
            re.search(ltp(direct_association), window_text["text_between"], flags=re.I),
            re.search(ltp(direct_association), left_window, flags=re.I),
            re.search(ltp(direct_association), right_window, flags=re.I),
        ]
    ):
        return POSITIVE
    else:
        return ABSTAIN


@labeling_function()
def LF_DaG_WEAK_ASSOCIATION(c):
    """
    This label function is design to search for phrases that indicate a
    weak association between the disease and gene
    """

    window_text = get_text_in_windows(c)
    left_window = " ".join(
        window_text["left_window"][0] + window_text["left_window"][1]
    )
    right_window = " ".join(
        window_text["right_window"][0] + window_text["right_window"][1]
    )

    if any(
        [
            re.search(ltp(weak_association), window_text["text_between"], flags=re.I),
            re.search(ltp(weak_association), left_window, flags=re.I),
            re.search(ltp(weak_association), right_window, flags=re.I),
        ]
    ):
        return POSITIVE
    else:
        return ABSTAIN


@labeling_function()
def LF_DaG_NO_ASSOCIATION(c):
    """
    This LF is designed to test if there is a key phrase that suggests
    a d-g pair is no an association.
    """
    window_text = get_text_in_windows(c)
    left_window = " ".join(
        window_text["left_window"][0] + window_text["left_window"][1]
    )
    right_window = " ".join(
        window_text["right_window"][0] + window_text["right_window"][1]
    )

    if re.search(ltp(no_direct_association), window_text["text_between"], flags=re.I):
        return NEGATIVE
    elif re.search(ltp(no_direct_association), left_window, flags=re.I):
        return NEGATIVE
    elif re.search(ltp(no_direct_association), right_window, flags=re.I):
        return NEGATIVE
    else:
        return ABSTAIN


@labeling_function()
def LF_DaG_CELLULAR_ACTIVITY(c):
    """
    This LF is designed to look for key phrases that indicate activity within a cell.
    e.x. positive immunostating for an experiment
    """
    window_text = get_text_in_windows(c)
    left_window = " ".join(
        window_text["left_window"][0] + window_text["left_window"][1]
    )
    right_window = " ".join(
        window_text["right_window"][0] + window_text["right_window"][1]
    )

    if any(
        [
            re.search(ltp(cellular_activity), window_text["text_between"], flags=re.I),
            re.search(ltp(cellular_activity), left_window, flags=re.I),
            re.search(ltp(cellular_activity), right_window, flags=re.I),
        ]
    ):
        return POSITIVE
    else:
        return ABSTAIN


@labeling_function()
def LF_DaG_DISEASE_SAMPLE(c):
    """
    This LF is designed to look for key phrases that indicate a sentence talking about tissue samples
    ex. cell line etc
    """
    window_text = get_text_in_windows(c)
    left_window = " ".join(
        window_text["left_window"][0] + window_text["left_window"][1]
    )
    right_window = " ".join(
        window_text["right_window"][0] + window_text["right_window"][1]
    )

    if any(
        [
            re.search(ltp(disease_sample_indicators), left_window, flags=re.I),
            re.search(ltp(disease_sample_indicators), right_window, flags=re.I),
        ]
    ):
        return POSITIVE
    else:
        return ABSTAIN


@labeling_function()
def LF_DG_METHOD_DESC(c):
    """
    This label function is designed to look for phrases
    that imply a sentence is description an experimental design
    """
    # Look at beginning of the sentence to see if
    # sentence is about experimental design
    sentence_tokens = " ".join(c.word[0:20])
    if re.search(ltp(method_indication), sentence_tokens, flags=re.I):
        return NEGATIVE
    else:
        return ABSTAIN


@labeling_function()
def LF_DG_TITLE(c):
    """
    This label function is designed to look for phrases that inditcates
    a paper title
    """
    beginning_text = " ".join(c.word[0:10])
    window_text = get_text_in_windows(c)

    if any(
        [
            re.search(
                r"^(\[|\[ )?" + ltp(title_indication), beginning_text, flags=re.I
            ),
            re.search(ltp(title_indication) + r"$", beginning_text, flags=re.I),
            re.search(r"author\'s transl", c.text, flags=re.I),
            ":" in " ".join(window_text["text_between"]),
        ]
    ):
        return NEGATIVE
    else:
        return ABSTAIN


@labeling_function()
def LF_DG_GENETIC_ABNORMALITIES(c):
    """
    This LF searches for key phraes that indicate a genetic abnormality
    """
    window_text = get_text_in_windows(c)
    left_window = " ".join(
        window_text["left_window"][0] + window_text["left_window"][1]
    )
    right_window = " ".join(
        window_text["right_window"][0] + window_text["right_window"][1]
    )

    if any(
        [
            re.search(
                ltp(genetic_abnormalities), window_text["text_between"], flags=re.I
            ),
            re.search(ltp(genetic_abnormalities), left_window, flags=re.I),
            re.search(ltp(genetic_abnormalities), right_window, flags=re.I),
        ]
    ):
        return POSITIVE
    return ABSTAIN


@labeling_function()
def LF_DG_DIAGNOSIS(c):
    """
    This label function is designed to search for words that imply a patient diagnosis
    which will provide evidence for possible disease gene association.
    """
    window_text = get_text_in_windows(c)
    tagged_text = get_tagged_text(
        c.word, c.disease_start, c.disease_end, c.gene_start, c.gene_end
    )

    if any(
        [
            re.search(
                r".*" + ltp(diagnosis_indicators) + r".*",
                window_text["text_between"],
                flags=re.I,
            ),
            re.search(
                r".*" + ltp(diagnosis_indicators) + r".*",
                window_text["text_between"],
                flags=re.I,
            ),
            re.search(
                r"({{A}}|{{B}}).*({{A}}|{{B}}).*" + ltp(diagnosis_indicators),
                tagged_text,
                flags=re.I,
            ),
        ]
    ):
        return POSITIVE
    else:
        return ABSTAIN


@labeling_function()
def LF_DG_PATIENT_WITH(c):
    """
    This label function looks for the phrase "  with" disease.
    """
    tagged_text = get_tagged_text(
        c.word, c.disease_start, c.disease_end, c.gene_start, c.gene_end
    )

    if re.search(r"patient(s)? with.{1,200}{{A}}", tagged_text, flags=re.I):
        return POSITIVE
    else:
        return ABSTAIN


@labeling_function()
def LF_DG_CONCLUSION_TITLE(c):
    """ "
    This label function searches for the word conclusion at the beginning of the sentence.
    Some abstracts are written in this format.
    """
    tagged_text = get_tagged_text(
        c.word, c.disease_start, c.disease_end, c.gene_start, c.gene_end
    )

    if "CONCLUSION:" in tagged_text or "concluded" in tagged_text:
        return POSITIVE
    else:
        return ABSTAIN


@labeling_function()
def LF_DaG_NO_CONCLUSION(c):
    """
    This label function fires a -1 if the number of negative label functinos is greater than the number
    of positive label functions.
    The main idea behind this label function is add support to sentences that could
    mention a possible disease gene association.
    """
    positive_num = np.sum(
        [
            LF_DaG_ASSOCIATION(c) == POSITIVE,
            LF_DG_IS_BIOMARKER(c) == POSITIVE,
            LF_DG_DIAGNOSIS(c) == POSITIVE,
            LF_DaG_CELLULAR_ACTIVITY(c) == POSITIVE,
            LF_DaG_WEAK_ASSOCIATION(c) == NEGATIVE,
            LF_DaG_NO_ASSOCIATION(c) == NEGATIVE,
        ]
    )
    negative_num = np.abs(
        np.sum(
            [
                LF_DG_METHOD_DESC(c) == NEGATIVE,
                LF_DG_TITLE(c) == NEGATIVE,
                LF_DG_NO_VERB(c) == NEGATIVE,
            ]
        )
    )
    if positive_num - negative_num >= 1:
        return ABSTAIN
    return NEGATIVE


@labeling_function()
def LF_DaG_CONCLUSION(c):
    """
    This label function fires a 1 if the number of positive label functions is greater than the number
    of negative label functions.
    The main idea behind this label function is add support to sentences that could
    mention a possible disease gene association
    """
    if not LF_DaG_NO_CONCLUSION(c) == NEGATIVE:
        if (
            LF_DaG_WEAK_ASSOCIATION(c) == NEGATIVE
            or LF_DaG_NO_ASSOCIATION(c) == NEGATIVE
        ):
            return NEGATIVE
        return POSITIVE

    else:
        return ABSTAIN


@labeling_function()
def LF_DG_DISTANCE_SHORT(c):
    """
    This LF is designed to make sure that the disease mention
    and the gene mention aren't right next to each other.
    """
    window_text = get_text_in_windows(c)
    return NEGATIVE if len(window_text["text_between"].split(" ")) <= 2 else ABSTAIN


@labeling_function()
def LF_DG_DISTANCE_LONG(c):
    """
    This LF is designed to make sure that the disease mention
    and the gene mention aren't too far from each other.
    """
    window_text = get_text_in_windows(c)
    return NEGATIVE if len(window_text["text_between"].split(" ")) > 50 else ABSTAIN


@labeling_function()
def LF_DG_ALLOWED_DISTANCE(c):
    """
    This LF is designed to make sure that the disease mention
    and the gene mention are in an acceptable distance between
    each other
    """
    return (
        ABSTAIN
        if any(
            [LF_DG_DISTANCE_LONG(c) == NEGATIVE, LF_DG_DISTANCE_SHORT(c) == NEGATIVE]
        )
        else POSITIVE
    )


@labeling_function()
def LF_DG_NO_VERB(c):
    """
    This label function is designed to fire if a given
    sentence doesn't contain a verb. Helps cut out some of the titles
    hidden in Pubtator abstracts
    """
    tags = list(filter(lambda x: "VB" in x and x != "VBG", c.pos_tag))
    if len(tags) == 0:
        return NEGATIVE
    return ABSTAIN


@labeling_function()
def LF_DG_CONTEXT_SWITCH(c):
    window_text = get_text_in_windows(c)
    if re.search(ltp(context_change_keywords), window_text["text_between"], flags=re.I):
        return NEGATIVE
    return ABSTAIN


"""
Bi-Clustering LFs
"""
path = (
    pathlib.Path(__file__)
    .joinpath("../../dependency_cluster/disease_gene_bicluster_results.tsv.xz")
    .resolve()
)
bicluster_dep_df = pd.read_csv(path, sep="\t")
causal_mutations_base = set(
    [
        tuple(x)
        for x in bicluster_dep_df.query("U>0")[["pubmed_id", "sentence_num"]].values
    ]
)
mutations_base = set(
    [
        tuple(x)
        for x in bicluster_dep_df.query("Ud>0")[["pubmed_id", "sentence_num"]].values
    ]
)
drug_targets_base = set(
    [
        tuple(x)
        for x in bicluster_dep_df.query("D>0")[["pubmed_id", "sentence_num"]].values
    ]
)
pathogenesis_base = set(
    [
        tuple(x)
        for x in bicluster_dep_df.query("J>0")[["pubmed_id", "sentence_num"]].values
    ]
)
therapeutic_base = set(
    [
        tuple(x)
        for x in bicluster_dep_df.query("Te>0")[["pubmed_id", "sentence_num"]].values
    ]
)
polymorphisms_base = set(
    [
        tuple(x)
        for x in bicluster_dep_df.query("Y>0")[["pubmed_id", "sentence_num"]].values
    ]
)
progression_base = set(
    [
        tuple(x)
        for x in bicluster_dep_df.query("G>0")[["pubmed_id", "sentence_num"]].values
    ]
)
biomarkers_base = set(
    [
        tuple(x)
        for x in bicluster_dep_df.query("Md>0")[["pubmed_id", "sentence_num"]].values
    ]
)
overexpression_base = set(
    [
        tuple(x)
        for x in bicluster_dep_df.query("X>0")[["pubmed_id", "sentence_num"]].values
    ]
)
regulation_base = set(
    [
        tuple(x)
        for x in bicluster_dep_df.query("L>0")[["pubmed_id", "sentence_num"]].values
    ]
)


@labeling_function()
def LF_DG_BICLUSTER_CASUAL_MUTATIONS(c):
    """
    This label function uses the bicluster data located in the
    A global network of biomedical relationships
    """
    sen_pos = c.get_parent().position
    pubmed_id = int(c.get_parent().document.name)
    if (pubmed_id, sen_pos) in causal_mutations_base:
        return POSITIVE
    return ABSTAIN


@labeling_function()
def LF_DG_BICLUSTER_MUTATIONS(c):
    """
    This label function uses the bicluster data located in the
    A global network of biomedical relationships
    """
    sen_pos = c.get_parent().position
    pubmed_id = int(c.get_parent().document.name)
    if (pubmed_id, sen_pos) in mutations_base:
        return POSITIVE
    return ABSTAIN


@labeling_function()
def LF_DG_BICLUSTER_DRUG_TARGETS(c):
    """
    This label function uses the bicluster data located in the
    A global network of biomedical relationships
    """
    sen_pos = c.get_parent().position
    pubmed_id = int(c.get_parent().document.name)
    if (pubmed_id, sen_pos) in drug_targets_base:
        return POSITIVE
    return ABSTAIN


@labeling_function()
def LF_DG_BICLUSTER_PATHOGENESIS(c):
    """
    This label function uses the bicluster data located in the
    A global network of biomedical relationships
    """
    sen_pos = c.get_parent().position
    pubmed_id = int(c.get_parent().document.name)
    if (pubmed_id, sen_pos) in pathogenesis_base:
        return POSITIVE
    return ABSTAIN


@labeling_function()
def LF_DG_BICLUSTER_THERAPEUTIC(c):
    """
    This label function uses the bicluster data located in the
    A global network of biomedical relationships
    """
    sen_pos = c.get_parent().position
    pubmed_id = int(c.get_parent().document.name)
    if (pubmed_id, sen_pos) in therapeutic_base:
        return POSITIVE
    return ABSTAIN


@labeling_function()
def LF_DG_BICLUSTER_POLYMORPHISMS(c):
    """
    This label function uses the bicluster data located in the
    A global network of biomedical relationships
    """
    sen_pos = c.get_parent().position
    pubmed_id = int(c.get_parent().document.name)
    if (pubmed_id, sen_pos) in polymorphisms_base:
        return POSITIVE
    return ABSTAIN


@labeling_function()
def LF_DG_BICLUSTER_PROGRESSION(c):
    """
    This label function uses the bicluster data located in the
    A global network of biomedical relationships
    """
    sen_pos = c.get_parent().position
    pubmed_id = int(c.get_parent().document.name)
    if (pubmed_id, sen_pos) in progression_base:
        return POSITIVE
    return ABSTAIN


@labeling_function()
def LF_DG_BICLUSTER_BIOMARKERS(c):
    """
    This label function uses the bicluster data located in the
    A global network of biomedical relationships
    """
    sen_pos = c.get_parent().position
    pubmed_id = int(c.get_parent().document.name)
    if (pubmed_id, sen_pos) in biomarkers_base:
        return POSITIVE
    return ABSTAIN


@labeling_function()
def LF_DG_BICLUSTER_OVEREXPRESSION(c):
    """
    This label function uses the bicluster data located in the
    A global network of biomedical relationships
    """
    sen_pos = c.get_parent().position
    pubmed_id = int(c.get_parent().document.name)
    if (pubmed_id, sen_pos) in overexpression_base:
        return POSITIVE
    return ABSTAIN


@labeling_function()
def LF_DG_BICLUSTER_REGULATION(c):
    """
    This label function uses the bicluster data located in the
    A global network of biomedical relationships
    """
    sen_pos = c.get_parent().position
    pubmed_id = int(c.get_parent().document.name)
    if (pubmed_id, sen_pos) in regulation_base:
        return POSITIVE
    return ABSTAIN


"""
RETRUN LFs to Notebook
"""

LFS = OrderedDict(
    {
        "distant_supervision": {
            "LF_HETNET_DISEASES": LF_HETNET_DISEASES,
            "LF_HETNET_DOAF": LF_HETNET_DOAF,
            "LF_HETNET_DisGeNET": LF_HETNET_DisGeNET,
            "LF_HETNET_GWAS": LF_HETNET_GWAS,
            "LF_HETNET_DaG_ABSENT": LF_HETNET_DaG_ABSENT,
        },
        "text_patterns": {
            "LF_DG_IS_BIOMARKER": LF_DG_IS_BIOMARKER,
            "LF_DaG_ASSOCIATION": LF_DaG_ASSOCIATION,
            "LF_DaG_WEAK_ASSOCIATION": LF_DaG_WEAK_ASSOCIATION,
            "LF_DaG_NO_ASSOCIATION": LF_DaG_NO_ASSOCIATION,
            "LF_DaG_CELLULAR_ACTIVITY": LF_DaG_CELLULAR_ACTIVITY,
            "LF_DaG_DISEASE_SAMPLE": LF_DaG_DISEASE_SAMPLE,
            "LF_DG_METHOD_DESC": LF_DG_METHOD_DESC,
            "LF_DG_TITLE": LF_DG_TITLE,
            "LF_DG_GENETIC_ABNORMALITIES": LF_DG_GENETIC_ABNORMALITIES,
            "LF_DG_DIAGNOSIS": LF_DG_DIAGNOSIS,
            "LF_DG_PATIENT_WITH": LF_DG_PATIENT_WITH,
            "LF_DG_CONCLUSION_TITLE": LF_DG_CONCLUSION_TITLE,
            "LF_DaG_NO_CONCLUSION": LF_DaG_NO_CONCLUSION,
            "LF_DaG_CONCLUSION": LF_DaG_CONCLUSION,
            "LF_DG_DISTANCE_SHORT": LF_DG_DISTANCE_SHORT,
            "LF_DG_DISTANCE_LONG": LF_DG_DISTANCE_LONG,
            "LF_DG_ALLOWED_DISTANCE": LF_DG_ALLOWED_DISTANCE,
            "LF_DG_NO_VERB": LF_DG_NO_VERB,
            "LF_DG_CONTEXT_SWITCH": LF_DG_CONTEXT_SWITCH,
        },
        "domain_heuristics": {
            "LF_DG_BICLUSTER_CASUAL_MUTATIONS": LF_DG_BICLUSTER_CASUAL_MUTATIONS,
            "LF_DG_BICLUSTER_MUTATIONS": LF_DG_BICLUSTER_MUTATIONS,
            "LF_DG_BICLUSTER_DRUG_TARGETS": LF_DG_BICLUSTER_DRUG_TARGETS,
            "LF_DG_BICLUSTER_PATHOGENESIS": LF_DG_BICLUSTER_PATHOGENESIS,
            "LF_DG_BICLUSTER_THERAPEUTIC": LF_DG_BICLUSTER_THERAPEUTIC,
            "LF_DG_BICLUSTER_POLYMORPHISMS": LF_DG_BICLUSTER_POLYMORPHISMS,
            "LF_DG_BICLUSTER_PROGRESSION": LF_DG_BICLUSTER_PROGRESSION,
            "LF_DG_BICLUSTER_BIOMARKERS": LF_DG_BICLUSTER_BIOMARKERS,
            "LF_DG_BICLUSTER_OVEREXPRESSION": LF_DG_BICLUSTER_OVEREXPRESSION,
            "LF_DG_BICLUSTER_REGULATION": LF_DG_BICLUSTER_REGULATION,
        },
    }
)
