# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.9.1+dev
#   kernelspec:
#     display_name: Python [conda env:snorkeling]
#     language: python
#     name: conda-env-snorkeling-py
# ---

# # Biclustering of Dependency Paths for Biomedical Realtionship Extraction

# A global network of biomedical relationships derived from text

# +
import networkx as nx
import pandas as pd
import spacy
from tqdm import tqdm

tqdm.pandas(desc="LF progressbar")
# -

nlp = spacy.load("en_core_web_sm")


def convert_dep_path(dataframe_row):
    is_first_entity_first = int(
        dataframe_row["first_entity_location"].split(",")[1]
    ) < int(dataframe_row["second_entity_location"].split(",")[0])
    seen_first_entity = False
    build_string = []
    for token in dataframe_row.sentence.split(" "):
        if token == dataframe_row["second_entity_name"]:
            if is_first_entity_first and seen_first_entity:
                token = "end_entity"

            if not is_first_entity_first:
                token = "start_entity"

        if token == dataframe_row["first_entity_name"]:
            if is_first_entity_first:
                token = "start_entity"
                seen_first_entity = True
            else:
                token = "end_entity"

        build_string.append(token.lower())

    # Skip sentences with multiple entities
    sen_text = " ".join(build_string).lower()
    if sen_text.count("start_entity") > 1 or sen_text.count("end_entity") > 1:
        return ""

    try:
        doc = nlp(sen_text)
        dep_graph = nx.Graph()
        for token in doc:
            for child in token.children:
                dep_graph.add_edge(
                    "{0}".format(token), "{0}".format(child), dep=child.dep_
                )

        path = nx.shortest_path(dep_graph, "start_entity", "end_entity")
        pathGraph = nx.path_graph(path)

        return " ".join(
            [
                f"{ea[1]}|{dep_graph.edges[ea[0], ea[1]]['dep']}|{ea[0]}"
                for ea in pathGraph.edges()
            ]
        )

    except nx.NetworkXNoPath:
        return ""
    except nx.NodeNotFound:
        return ""


# # Chemical-Disease

chemical_disease_url = "https://zenodo.org/record/1495808/files/part-i-chemical-disease-path-theme-distributions.txt.zip"
chemical_disease_paths_url = "https://zenodo.org/record/1495808/files/part-ii-dependency-paths-chemical-disease-sorted-with-themes.txt.zip"

chemical_disease_path_dist_df = pd.read_table(chemical_disease_url)
chemical_disease_path_dist_df.head(2)

chemical_disease_paths_df = pd.read_table(
    chemical_disease_paths_url,
    names=[
        "pubmed_id",
        "sentence_num",
        "first_entity_name",
        "first_entity_location",
        "second_entity_name",
        "second_entity_location",
        "first_entity_name_raw",
        "second_entity_name_raw",
        "first_entity_db_id",
        "second_entity_db_id",
        "first_entity_type",
        "second_entity_type",
        "dep_path",
        "sentence",
    ],
)
chemical_disease_paths_df.head(2)

chemical_disease_merged_path_df = chemical_disease_paths_df.assign(
    dep_path=chemical_disease_paths_df.dep_path.apply(lambda x: x.lower()).values,
    spacy_dep_path=lambda x: x.progress_apply(convert_dep_path, axis=1),
).merge(
    chemical_disease_path_dist_df.rename(index=str, columns={"path": "dep_path"}),
    on=["dep_path"],
)
chemical_disease_merged_path_df.head(2)

chemical_disease_merged_path_df.to_csv(
    "chemical_disease_bicluster_results.tsv.xz", sep="\t", index=False, compression="xz"
)

# # Chemical-Gene

chemical_gene_url = "https://zenodo.org/record/1495808/files/part-i-chemical-gene-path-theme-distributions.txt.zip"
chemical_gene_paths_url = "https://zenodo.org/record/1495808/files/part-ii-dependency-paths-chemical-gene-sorted-with-themes.txt.zip"

chemical_gene_path_dist_df = pd.read_table(chemical_gene_url)
chemical_gene_path_dist_df.head(2)

chemical_gene_paths_df = pd.read_table(
    chemical_gene_paths_url,
    names=[
        "pubmed_id",
        "sentence_num",
        "first_entity_name",
        "first_entity_location",
        "second_entity_name",
        "second_entity_location",
        "first_entity_name_raw",
        "second_entity_name_raw",
        "first_entity_db_id",
        "second_entity_db_id",
        "first_entity_type",
        "second_entity_type",
        "dep_path",
        "sentence",
    ],
)
chemical_gene_paths_df.head(2)

chemical_gene_merged_path_df = chemical_gene_paths_df.assign(
    dep_path=chemical_gene_paths_df.dep_path.apply(lambda x: x.lower()).values,
    spacy_dep_path=lambda x: x.progress_apply(convert_dep_path, axis=1),
).merge(
    chemical_gene_path_dist_df.rename(index=str, columns={"path": "dep_path"}),
    on=["dep_path"],
)
chemical_gene_merged_path_df.head(2)

chemical_gene_merged_path_df.to_csv(
    "chemical_gene_bicluster_results.tsv.xz", sep="\t", index=False, compression="xz"
)

# # Disease-Gene

disease_gene_url = "https://zenodo.org/record/1495808/files/part-i-gene-disease-path-theme-distributions.txt.zip"
disease_gene_paths_url = "https://zenodo.org/record/1495808/files/part-ii-dependency-paths-gene-disease-sorted-with-themes.txt.zip"

disease_gene_path_dist_df = pd.read_table(disease_gene_url)
disease_gene_path_dist_df.head(2)

disease_gene_paths_df = pd.read_table(
    disease_gene_paths_url,
    names=[
        "pubmed_id",
        "sentence_num",
        "first_entity_name",
        "first_entity_location",
        "second_entity_name",
        "second_entity_location",
        "first_entity_name_raw",
        "second_entity_name_raw",
        "first_entity_db_id",
        "second_entity_db_id",
        "first_entity_type",
        "second_entity_type",
        "dep_path",
        "sentence",
    ],
)
disease_gene_paths_df.head(2)

disease_gene_merged_path_df = disease_gene_paths_df.assign(
    dep_path=disease_gene_paths_df.dep_path.apply(lambda x: x.lower()).values,
    spacy_dep_path=lambda x: x.progress_apply(convert_dep_path, axis=1),
).merge(
    disease_gene_path_dist_df.rename(index=str, columns={"path": "dep_path"}),
    on=["dep_path"],
)
disease_gene_merged_path_df.head(2)

disease_gene_merged_path_df.to_csv(
    "disease_gene_bicluster_results.tsv.xz", sep="\t", index=False, compression="xz"
)

# # Gene-Gene

gene_gene_url = "https://zenodo.org/record/1495808/files/part-i-gene-gene-path-theme-distributions.txt.zip"
gene_gene_paths_url = "https://zenodo.org/record/1495808/files/part-ii-dependency-paths-gene-gene-sorted-with-themes.txt.zip"

gene_gene_path_dist_df = pd.read_table(gene_gene_url)
gene_gene_path_dist_df.head(2)

gene_gene_paths_df = pd.read_table(
    gene_gene_paths_url,
    names=[
        "pubmed_id",
        "sentence_num",
        "first_entity_name",
        "first_entity_location",
        "second_entity_name",
        "second_entity_location",
        "first_entity_name_raw",
        "second_entity_name_raw",
        "first_entity_db_id",
        "second_entity_db_id",
        "first_entity_type",
        "second_entity_type",
        "dep_path",
        "sentence",
    ],
)
gene_gene_paths_df.head(2)

gene_gene_merged_path_df = gene_gene_paths_df.assign(
    dep_path=gene_gene_paths_df.dep_path.apply(lambda x: x.lower()).values,
    spacy_dep_path=lambda x: x.progress_apply(convert_dep_path, axis=1),
).merge(
    gene_gene_path_dist_df.rename(index=str, columns={"path": "dep_path"}),
    on=["dep_path"],
)
gene_gene_merged_path_df.head(2)

gene_gene_merged_path_df.to_csv(
    "gene_gene_bicluster_results.tsv.xz", sep="\t", index=False, compression="xz"
)
