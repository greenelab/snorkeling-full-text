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
    .joinpath("../../knowledge_bases/compound_binds_gene.tsv.xz")
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
        key = str(row.entrez_gene_id), row.drugbank_id, source
        knowledge_base.add(key)


@labeling_function()
def LF_HETNET_DRUGBANK(c):
    """
    This label function returns 1 if the given Disease Gene pair is
    located in the Drugbank database
    """
    return (
        POSITIVE
        if (c.gene_cid, c.compound_cid, "DrugBank") in knowledge_base
        else ABSTAIN
    )


@labeling_function()
def LF_HETNET_DRUGCENTRAL(c):
    """
    This label function returns 1 if the given Disease Gene pair is
    located in the Drugcentral database
    """
    return (
        POSITIVE
        if (c.gene_cid, c.compound_cid, "DrugCentral") in knowledge_base
        else ABSTAIN
    )


@labeling_function()
def LF_HETNET_ChEMBL(c):
    """
    This label function returns 1 if the given Disease Gene pair is
    located in the ChEMBL database
    """
    return (
        POSITIVE
        if (c.gene_cid, c.compound_cid, "ChEMBL") in knowledge_base
        else ABSTAIN
    )


@labeling_function()
def LF_HETNET_BINDINGDB(c):
    """
    This label function returns 1 if the given Disease Gene pair is
    located in the BindingDB database
    """
    return (
        POSITIVE
        if (c.gene_cid, c.compound_cid, "BindingDB") in knowledge_base
        else ABSTAIN
    )


@labeling_function()
def LF_HETNET_PDSP_KI(c):
    """
    This label function returns 1 if the given Disease Gene pair is
    located in the PDSP_KI database
    """
    return (
        POSITIVE
        if (c.gene_cid, c.compound_cid, "PDSP Ki") in knowledge_base
        else ABSTAIN
    )


@labeling_function()
def LF_HETNET_US_PATENT(c):
    """
    This label function returns 1 if the given Disease Gene pair is
    located in the US PATENT database
    """
    return (
        POSITIVE
        if (c.gene_cid, c.compound_cid, "US Patent") in knowledge_base
        else ABSTAIN
    )


@labeling_function()
def LF_HETNET_PUBCHEM(c):
    """
    This label function returns 1 if the given Disease Gene pair is
    located in the PUBCHEM database
    """
    return (
        POSITIVE
        if (c.gene_cid, c.compound_cid, "PubChem") in knowledge_base
        else ABSTAIN
    )


@labeling_function()
def LF_HETNET_CG_ABSENT(c):
    """
    This label function fires -1 if the given Compound Disease pair does not appear
    in the databases above.
    """
    return (
        ABSTAIN
        if any(
            [
                LF_HETNET_DRUGBANK(c) == POSITIVE,
                LF_HETNET_DRUGCENTRAL(c) == POSITIVE,
                LF_HETNET_ChEMBL(c) == POSITIVE,
                LF_HETNET_BINDINGDB(c) == POSITIVE,
                LF_HETNET_PDSP_KI(c) == POSITIVE,
                LF_HETNET_US_PATENT(c) == POSITIVE,
                LF_HETNET_PUBCHEM(c) == POSITIVE,
            ]
        )
        else NEGATIVE
    )


# obtained from ftp://ftp.ncbi.nlm.nih.gov/gene/DATA/ (ncbi's ftp server)
# https://github.com/dhimmel/entrez-gene/blob/a7362748a34211e5df6f2d185bb3246279760546/download/Homo_sapiens.gene_info.gz <-- use pandas and trim i guess
columns = [
    "tax_id",
    "GeneID",
    "Symbol",
    "LocusTag",
    "Synonyms",
    "dbXrefs",
    "chromosome",
    "map_location",
    "description",
    "type_of_gene",
    "Symbol_from_nomenclature_authority",
    "Full_name_from_nomenclature_authority",
    "Nomenclature_status",
    "Other_designations",
    "Modification_date",
]
gene_desc = pd.read_table(
    "https://github.com/dhimmel/entrez-gene/blob/a7362748a34211e5df6f2d185bb3246279760546/download/Homo_sapiens.gene_info.gz?raw=true",
    sep="\t",
    names=columns,
    compression="gzip",
    skiprows=1,
)


@labeling_function()
def LF_CG_CHECK_GENE_TAG(c):
    """
    This label function is used for labeling each passed candidate as either pos or neg.
    Keyword Args:
    c- the candidate object to be passed in.
    """

    gene_name = " ".join(c.word[c.gene_start : c.gene_end])
    gene_entry_df = gene_desc.query("GeneID == @gene_id")

    if gene_entry_df.empty:
        return NEGATIVE

    for token in gene_name.split(" "):
        if (
            gene_entry_df["Symbol"].values[0].lower() == token
            or token in gene_entry_df["Synonyms"].values[0].lower()
        ):
            return ABSTAIN
        elif token in gene_entry_df["description"].values[0].lower():
            return ABSTAIN
    return NEGATIVE


"""
SENTENCE PATTERN MATCHING
"""

binding_indication = {
    "binding",
    "binding site of",
    "heterodimerizes with",
    "reaction of",
    "binding of",
    "effects on",
    "by an inhibitor",
    "stimulated by",
    "reaction of",
    "can activate",
    "infusion of",
    "inhibited by",
    "receptor binding",
    "inhibitor(s)? of",
    "kinase inhibitors" "interaction of",
    "phosphorylation",
    "interacts with",
    "agonistic",
    "oxidation of",
    "oxidized to",
    "attack of",
}

weak_binding_indications = {
    "affected the activity of",
    "catalytic activity",
    "intermolecular interactions",
    "possible target protein",
    "local membraine effects",
    "response(s)? to",
}


upregulates = {
    "enhanced",
    "increased expression",
    "reversed the decreased.*(response)?",
    "maximial activation of",
    "increased expression of",
    "augmented",
    r"\bhigh\b",
    "elevate(d|s)?",
    "(significant(ly)?)? increase(d|s)?",
    "greated for",
    "greater in",
    "higher",
    "prevent their degeneration",
    "activate",
    "evoked a sustained rise",
}

downregulates = {
    "regulate transcription of",
    "inhibitors of",
    "kinase inhibitors",
    "negatively regulated by",
    "inverse agonist of",
    "downregulated",
    "suppressed",
    "\blow\b",
    "reduce(d|s)?",
    "(significant(ly)?)? decrease(d|s)?",
    "inhibited by",
    "not higher",
    "unresponsive",
    "reduce",
    "antagonist",
    "inhibit(or|its)",
    "significantly attenuated",
}

gene_receivers = {
    "receptor",
    "(protein )?kinase",
    "antagonist",
    "agonist",
    "subunit",
    "binding",
    "bound",
}

compound_indentifiers = {"small molecules", "inhibitor"}


@labeling_function()
def LF_CG_BINDING(c):
    """
    This label function is designed to look for phrases
    that imply a compound binding to a gene/protein
    """
    window_text = get_text_in_windows(c)
    compound_left_window = " ".join(window_text["left_window"][0])
    compound_right_window = " ".join(window_text["right_window"][0])

    if any(
        [
            re.search(ltp(binding_indication), window_text["text_between"], flags=re.I),
            re.search(ltp(binding_indication), compound_left_window, flags=re.I),
            re.search(ltp(binding_indication), compound_right_window, flags=re.I),
        ]
    ):
        return POSITIVE

    return ABSTAIN


@labeling_function()
def LF_CG_WEAK_BINDING(c):
    """
    This label function is designed to look for phrases
    that could imply a compound binding to a gene/protein
    """
    window_text = get_text_in_windows(c)
    if re.search(
        ltp(weak_binding_indications), window_text["text_between"], flags=re.I
    ):
        return POSITIVE

    return ABSTAIN


@labeling_function()
def LF_CG_UPREGULATES(c):
    """
    This label function is designed to look for phrases
    that implies a compound increaseing activity of a gene/protein
    """
    window_text = get_text_in_windows(c, window_size=2)
    compound_left_window = " ".join(window_text["left_window"][0])

    if any(
        [
            re.search(ltp(upregulates), window_text["text_between"], flags=re.I),
            re.search(ltp(upregulates), compound_left_window),
        ]
    ):
        return POSITIVE

    return ABSTAIN


@labeling_function()
def LF_CG_DOWNREGULATES(c):
    """
    This label function is designed to look for phrases
    that could implies a compound decreasing the activity of a gene/protein
    """
    window_text = get_text_in_windows(c, window_size=2)
    compound_left_window = " ".join(window_text["left_window"][0])

    if any(
        [
            re.search(ltp(downregulates), window_text["text_between"], flags=re.I),
            re.search(ltp(downregulates), compound_left_window),
        ]
    ):
        return POSITIVE

    return ABSTAIN


@labeling_function()
def LF_CG_GENE_RECEIVERS(c):
    """
    This label function is designed to look for phrases
    that imples a kinases or sort of protein that receives
    a stimulus to function
    """
    window_text = get_text_in_windows(c, window_size=4)
    gene_left_window = " ".join(window_text["left_window"][1])
    gene_right_window = " ".join(window_text["right_window"][1])
    gene_start = window_text["entity_columns"]["entity_two_start"]
    gene_end = window_text["entity_columns"]["entity_two_end"]

    if any(
        [
            re.search(
                ltp(gene_receivers),
                " ".join(c.word[c[gene_start] : c[gene_end]]),
                flags=re.I,
            ),
            re.search(ltp(gene_receivers), gene_left_window, flags=re.I),
            re.search(ltp(gene_receivers), gene_right_window, flags=re.I),
        ]
    ):
        return POSITIVE

    return ABSTAIN


@labeling_function()
def LF_CG_ASE_SUFFIX(c):
    """
    This label function is designed to look parts of the gene tags
    that implies a sort of "ase" or enzyme
    """
    window_text = get_text_in_windows(c, window_size=4)
    gene_start = window_text["entity_columns"]["entity_two_start"]
    gene_end = window_text["entity_columns"]["entity_two_end"]

    if re.search(r"ase\b", " ".join(c.word[c[gene_start] : c[gene_end]]), flags=re.I):
        return POSITIVE

    return ABSTAIN


@labeling_function()
def LF_CG_IN_SERIES(c):
    """
    This label function is designed to look for a mention being caught
    in a series of other genes or compounds
    """
    if len(re.findall(r",", c.text)) >= 2:
        if re.search(", and", c.text):
            return NEGATIVE
    return ABSTAIN


@labeling_function()
def LF_CG_ANTIBODY(c):
    """
    This label function is designed to look for phrase
    antibody.
    """
    window_text = get_text_in_windows(c, window_size=3)
    gene_right_window = " ".join(window_text["right_window"][1])

    gene_start = window_text["entity_columns"]["entity_two_start"]
    gene_end = window_text["entity_columns"]["entity_two_end"]

    if any(
        [
            re.search(
                "antibod(y|ies)",
                " ".join(c.word[c[gene_start] : c[gene_end]]),
                flags=re.I,
            ),
            re.search("antibod(y|ies)", gene_right_window),
        ]
    ):
        return POSITIVE

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
    "Regulation of",
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
}


@labeling_function()
def LF_CG_METHOD_DESC(c):
    """
    This label function is designed to look for phrases
    that imply a sentence is description an experimental design
    """
    if re.search(ltp(method_indication), c.text, flags=re.I):
        return NEGATIVE

    return ABSTAIN


@labeling_function()
def LF_CG_NO_CONCLUSION(c):
    """
    This label function fires a -1 if the number of negative label functinos is greater than the number
    of positive label functions.
    The main idea behind this label function is add support to sentences that could
    mention a possible disease gene association.
    """
    positive_num = np.sum(
        [
            LF_CG_BINDING(c) == POSITIVE,
            LF_CG_WEAK_BINDING(c) == POSITIVE,
            LF_CG_GENE_RECEIVERS(c) == POSITIVE,
            LF_CG_ANTIBODY(c) == POSITIVE,
        ]
    )

    negative_num = np.sum(LF_CG_METHOD_DESC(c) == NEGATIVE)

    if positive_num - negative_num >= 1:
        return ABSTAIN

    return NEGATIVE


@labeling_function()
def LF_CG_CONCLUSION(c):
    """
    This label function fires a 1 if the number of positive label functions is greater than the number
    of negative label functions.
    The main idea behind this label function is add support to sentences that could
    mention a possible disease gene association
    """
    if not LF_CG_NO_CONCLUSION(c) == NEGATIVE:
        return POSITIVE

    return ABSTAIN


@labeling_function()
def LF_CG_DISTANCE_SHORT(c):
    """
    This LF is designed to make sure that the compound mention
    and the gene mention aren't right next to each other.
    """
    window_text = get_text_in_windows(c, window_size=3)

    return NEGATIVE if len(window_text["text_between"].split(" ")) <= 2 else ABSTAIN


@labeling_function()
def LF_CG_DISTANCE_LONG(c):
    """
    This LF is designed to make sure that the compound mention
    and the gene mention aren't too far from each other.
    """
    window_text = get_text_in_windows(c, window_size=3)
    return NEGATIVE if len(window_text["text_between"].split(" ")) > 25 else ABSTAIN


@labeling_function()
def LF_CG_ALLOWED_DISTANCE(c):
    """
    This LF is designed to make sure that the compound mention
    and the gene mention are in an acceptable distance between
    each other
    """
    return (
        ABSTAIN
        if any(
            [LF_CG_DISTANCE_LONG(c) == NEGATIVE, LF_CG_DISTANCE_SHORT(c) == NEGATIVE]
        )
        else POSITIVE
    )


@labeling_function()
def LF_CG_NO_VERB(c):
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


@labeling_function()
def LF_CG_PARENTHETICAL_DESC(c):
    """
    This label function looks for mentions that are in paranthesis.
    Some of the gene mentions are abbreviations rather than names of a gene.
    """
    window_text = get_text_in_windows(c, window_size=1)
    gene_left_window = window_text["left_window"][1]

    if all([")" in c.gene_span, "(" in gene_left_window, LF_CG_DISTANCE_SHORT(c)]):
        return NEGATIVE

    return ABSTAIN


"""
Bi-Clustering LFs
"""
path = (
    pathlib.Path(__file__)
    .joinpath("../../dependency_cluster/chemical_gene_bicluster_results.tsv.xz")
    .resolve()
)
cg_bicluster_dep_df = pd.read_csv(path, sep="\t")
cg_cat_codes = ["B", "A+", "A-", "E+", "E-", "E", "N"]
cg_dep_path_mapper = create_dep_mapper(cg_bicluster_dep_df, cg_cat_codes)


@labeling_function()
def LF_CG_BICLUSTER_BINDS(c):
    """
    This label function uses the bicluster data located in the
    A global network of biomedical relationships
    """
    return match_dep_path(c.dep_path, cg_dep_path_mapper, "B", POSITIVE, ABSTAIN)


@labeling_function()
def LF_CG_BICLUSTER_AGONISM(c):
    """
    This label function uses the bicluster data located in the
    A global network of biomedical relationships
    """
    return match_dep_path(c.dep_path, cg_dep_path_mapper, "A+", POSITIVE, ABSTAIN)


@labeling_function()
def LF_CG_BICLUSTER_ANTAGONISM(c):
    """
    This label function uses the bicluster data located in the
    A global network of biomedical relationships
    """
    return match_dep_path(c.dep_path, cg_dep_path_mapper, "A-", POSITIVE, ABSTAIN)


@labeling_function()
def LF_CG_BICLUSTER_INC_EXPRESSION(c):
    """
    This label function uses the bicluster data located in the
    A global network of biomedical relationships
    """
    return match_dep_path(c.dep_path, cg_dep_path_mapper, "E+", POSITIVE, ABSTAIN)


@labeling_function()
def LF_CG_BICLUSTER_DEC_EXPRESSION(c):
    """
    This label function uses the bicluster data located in the
    A global network of biomedical relationships
    """
    return match_dep_path(c.dep_path, cg_dep_path_mapper, "E-", POSITIVE, ABSTAIN)


@labeling_function()
def LF_CG_BICLUSTER_AFF_EXPRESSION(c):
    """
    This label function uses the bicluster data located in the
    A global network of biomedical relationships
    """
    return match_dep_path(c.dep_path, cg_dep_path_mapper, "E", POSITIVE, ABSTAIN)


@labeling_function()
def LF_CG_BICLUSTER_INHIBITS(c):
    """
    This label function uses the bicluster data located in the
    A global network of biomedical relationships
    """
    return match_dep_path(c.dep_path, cg_dep_path_mapper, "N", POSITIVE, ABSTAIN)


"""
RETRUN LFs to Notebook
"""

LFS = OrderedDict(
    {
        "distant_supervision": {
            "LF_HETNET_DRUGBANK": LF_HETNET_DRUGBANK,
            "LF_HETNET_DRUGCENTRAL": LF_HETNET_DRUGCENTRAL,
            "LF_HETNET_ChEMBL": LF_HETNET_ChEMBL,
            "LF_HETNET_BINDINGDB": LF_HETNET_BINDINGDB,
            "LF_HETNET_PDSP_KI": LF_HETNET_PDSP_KI,
            "LF_HETNET_US_PATENT": LF_HETNET_US_PATENT,
            "LF_HETNET_PUBCHEM": LF_HETNET_PUBCHEM,
            "LF_HETNET_CG_ABSENT": LF_HETNET_CG_ABSENT,
            "LF_CG_CHECK_GENE_TAG": LF_CG_CHECK_GENE_TAG,
        },
        "text_patterns": {
            "LF_CG_BINDING": LF_CG_BINDING,
            "LF_CG_WEAK_BINDING": LF_CG_WEAK_BINDING,
            "LF_CG_GENE_RECEIVERS": LF_CG_GENE_RECEIVERS,
            "LF_CG_ASE_SUFFIX": LF_CG_ASE_SUFFIX,
            "LF_CG_IN_SERIES": LF_CG_IN_SERIES,
            "LF_CG_ANTIBODY": LF_CG_ANTIBODY,
            "LF_CG_METHOD_DESC": LF_CG_METHOD_DESC,
            "LF_CG_NO_CONCLUSION": LF_CG_NO_CONCLUSION,
            "LF_CG_CONCLUSION": LF_CG_CONCLUSION,
            "LF_CG_DISTANCE_SHORT": LF_CG_DISTANCE_SHORT,
            "LF_CG_DISTANCE_LONG": LF_CG_DISTANCE_LONG,
            "LF_CG_ALLOWED_DISTANCE": LF_CG_ALLOWED_DISTANCE,
            "LF_CG_NO_VERB": LF_CG_NO_VERB,
        },
        "domain_heuristics": {
            "LF_CG_BICLUSTER_BINDS": LF_CG_BICLUSTER_BINDS,
            "LF_CG_BICLUSTER_AGONISM": LF_CG_BICLUSTER_AGONISM,
            "LF_CG_BICLUSTER_ANTAGONISM": LF_CG_BICLUSTER_ANTAGONISM,
            "LF_CG_BICLUSTER_INC_EXPRESSION": LF_CG_BICLUSTER_INC_EXPRESSION,
            "LF_CG_BICLUSTER_DEC_EXPRESSION": LF_CG_BICLUSTER_DEC_EXPRESSION,
            "LF_CG_BICLUSTER_AFF_EXPRESSION": LF_CG_BICLUSTER_AFF_EXPRESSION,
            "LF_CG_BICLUSTER_INHIBITS": LF_CG_BICLUSTER_INHIBITS,
        },
    }
)
