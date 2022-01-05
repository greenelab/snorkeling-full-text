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
#     display_name: Python [conda env:snorkeling_full_text]
#     language: python
#     name: conda-env-snorkeling_full_text-py
# ---

# # Disease Associates Genes Edge Prediction

# This notebook is designed to take the next step moving from predicted sentences to edge predictions. After training the discriminator model, each sentences contains a confidence score for the likelihood of mentioning a relationship. Multiple relationships contain multiple sentences, which makes establishing an edge unintuitive. Is taking the max score appropriate for determining existence of an edge? Does taking the mean of each relationship make more sense? The answer towards these questions are shown below.

# +
# %load_ext autoreload
# %autoreload 2
# %matplotlib inline

from pathlib import Path

import pandas as pd
import plydata as ply
import plydata.tidy as ply_tdy
from sqlalchemy import create_engine

# +
# Set up the environment
username = "danich1"
password = "snorkel"
dbname = "pubmed_central_db"

# Path subject to change for different os
database_str = (
    f"postgresql+psycopg2://{username}:{password}@/{dbname}?host=/var/run/postgresql"
)
conn = create_engine(database_str)
# -

# # Disease associates Gene

# ## Disease and Gene Info URLs for Hetionet

# +
disease_url = "https://raw.githubusercontent.com/dhimmel/disease-ontology/052ffcc960f5897a0575f5feff904ca84b7d2c1d/data/xrefs-prop-slim.tsv"
gene_url = "https://raw.githubusercontent.com/dhimmel/entrez-gene/a7362748a34211e5df6f2d185bb3246279760546/data/genes-human.tsv"
dag_url = "https://github.com/dhimmel/integrate/raw/93feba1765fbcd76fd79e22f25121f5399629148/compile/DaG-association.tsv"

# Not used but url for disease up/down regulates gene
# Leaving for future work if someone picks this project up
drg_url = "https://raw.githubusercontent.com/dhimmel/stargeo/08b126cc1f93660d17893c4a3358d3776e35fd84/data/diffex.tsv"
# -

# ## Load DataFrames for each URL

disease_ontology_df = (
    pd.read_csv(disease_url, sep="\t")
    >> ply.distinct(["doid_code", "doid_name"])
    >> ply.rename(doid_id="doid_code")
    >> ply.define(merge_key=1)
)
disease_ontology_df >> ply.slice_rows(5)

entrez_gene_df = (
    pd.read_csv(gene_url, sep="\t")
    >> ply.rename(dict(entrez_gene_id="GeneID", gene_symbol="Symbol"))
    >> ply.define(merge_key=1)
)
entrez_gene_df >> ply.slice_rows(5)

disease_gene_map_df = (
    entrez_gene_df
    >> ply.select("entrez_gene_id", "gene_symbol", "merge_key")
    >> ply.inner_join(
        disease_ontology_df >> ply.select("doid_id", "doid_name", "merge_key"),
        on="merge_key",
    )
    >> ply.select("-merge_key")
)
disease_gene_map_df >> ply.slice_rows(5)

hetionet_dag_df = pd.read_csv(
    dag_url, sep="\t", dtype={"entrez_gene_id": int}
) >> ply.define(merge_key=1)
hetionet_dag_df >> ply.slice_rows(5)

query = """
SELECT "disease_cid" AS doid_id, "gene_cid" AS  entrez_gene_id, count(*) AS n_sentences
FROM disease_gene
GROUP BY "disease_cid", "gene_cid"
"""
disease_gene_sentence_df = (
    pd.read_sql(query, conn)
    >> ply_tdy.separate_rows("entrez_gene_id", sep=";")
    >> ply.call(".astype", {"entrez_gene_id": int})
)
disease_gene_sentence_df >> ply.slice_rows(5)

# ## Merge all dataframes into One

disease_gene_associations_df = (
    disease_gene_map_df
    >> ply.left_join(
        hetionet_dag_df >> ply.select("doid_id", "entrez_gene_id", "sources"),
        on=["doid_id", "entrez_gene_id"],
    )
    >> ply.left_join(disease_gene_sentence_df, on=["doid_id", "entrez_gene_id"])
    >> ply.call(".fillna", {"n_sentences": 0})
    >> ply.call(".astype", {"n_sentences": int})
    >> ply.define(
        hetionet="sources.notnull().astype(int)",
        has_sentence="(n_sentences > 0).astype(int)",
    )
)
(disease_gene_associations_df >> ply.slice_rows(5) >> ply.call(".transpose"))

outfile = "disease_associates_gene.tsv.xz"
if not Path(outfile).exists():
    (
        disease_gene_associations_df
        >> ply.call(".to_csv", outfile, sep="\t", index=False, compression="xz")
    )

# free memory for rest of notebook
del disease_ontology_df
del entrez_gene_df
del disease_gene_map_df
del disease_gene_sentence_df
del hetionet_dag_df
del disease_gene_associations_df

# # Compound treats Disease

# ## Compound and Disease Info URLs for Hetionet

disease_url = "https://raw.githubusercontent.com/dhimmel/disease-ontology/052ffcc960f5897a0575f5feff904ca84b7d2c1d/data/xrefs-prop-slim.tsv"
compound_url = "https://raw.githubusercontent.com/dhimmel/drugbank/7b94454b14a2fa4bb9387cb3b4b9924619cfbd3e/data/drugbank.tsv"
ctpd_url = "https://raw.githubusercontent.com/dhimmel/indications/11d535ba0884ee56c3cd5756fdfb4985f313bd80/catalog/indications.tsv"

# ## Load DataFrames for each URL

disease_ontology_df = (
    pd.read_csv(disease_url, sep="\t")
    >> ply.distinct(["doid_code", "doid_name"])
    >> ply.rename(doid_id="doid_code")
    >> ply.define(merge_key=1)
)
disease_ontology_df >> ply.slice_rows(5)

drugbank_df = (
    pd.read_csv(compound_url, sep="\t")
    >> ply.rename(drug_name="name")
    >> ply.define(merge_key=1)
)
drugbank_df >> ply.slice_rows(5)

compound_disease_map_df = (
    drugbank_df
    >> ply.select("drugbank_id", "drug_name", "merge_key")
    >> ply.inner_join(
        disease_ontology_df >> ply.select("doid_id", "doid_name", "merge_key"),
        on="merge_key",
    )
    >> ply.select("-merge_key")
)
compound_disease_map_df >> ply.slice_rows(5)

hetionet_ctpd_df = (
    pd.read_csv(ctpd_url, sep="\t")
    >> ply.define(sources='"pharmacotherapydb"')
    >> ply.select("-n_curators", "-n_resources")
    >> ply.rename(dict(drug_name="drug", doid_name="disease"))
)
hetionet_ctpd_df >> ply.slice_rows(5)

query = """
SELECT "compound_cid" as drugbank_id, "disease_cid" as doid_id, count(*) AS n_sentences
FROM compound_disease
GROUP BY "compound_cid", "disease_cid";
"""
compound_disease_sentence_df = pd.read_sql(query, conn)
compound_disease_sentence_df >> ply.slice_rows(5)

# ## Merge all dataframes into One

compound_treats_disease_df = (
    compound_disease_map_df
    >> ply.left_join(
        hetionet_ctpd_df
        >> ply.query("category=='DM'")
        >> ply.select("doid_id", "drugbank_id", "category", "sources"),
        on=["drugbank_id", "doid_id"],
    )
    >> ply.left_join(compound_disease_sentence_df, on=["drugbank_id", "doid_id"])
    >> ply.call(".fillna", {"n_sentences": 0})
    >> ply.call(".astype", {"n_sentences": int})
    >> ply.define(
        hetionet="sources.notnull().astype(int)",
        has_sentence="(n_sentences > 0).astype(int)",
    )
)
compound_treats_disease_df >> ply.slice_rows(5)

outfile = "compound_treats_disease.tsv.xz"
if not Path(outfile).exists():
    (
        compound_treats_disease_df
        >> ply.call(".to_csv", outfile, sep="\t", index=False, compression="xz")
    )

# free memory for rest of notebook
del drugbank_df
del disease_ontology_df
del compound_disease_map_df
del compound_disease_sentence_df
del hetionet_ctpd_df
del compound_treats_disease_df

# # Compound binds Gene

# ## Compound and Gene Info URLs for Hetionet

compound_url = "https://raw.githubusercontent.com/dhimmel/drugbank/7b94454b14a2fa4bb9387cb3b4b9924619cfbd3e/data/drugbank.tsv"
gene_url = "https://raw.githubusercontent.com/dhimmel/entrez-gene/a7362748a34211e5df6f2d185bb3246279760546/data/genes-human.tsv"
cbg_url = "https://raw.githubusercontent.com/dhimmel/integrate/93feba1765fbcd76fd79e22f25121f5399629148/compile/CbG-binding.tsv"

# ## Load DataFrames for each URL

entrez_gene_df = (
    pd.read_csv(gene_url, sep="\t")
    >> ply.rename(dict(entrez_gene_id="GeneID", gene_symbol="Symbol"))
    >> ply.define(merge_key=1)
)
entrez_gene_df >> ply.slice_rows(5)

drugbank_df = (
    pd.read_csv(compound_url, sep="\t")
    >> ply.rename(dict(drug_name="name"))
    >> ply.define(merge_key=1)
)
drugbank_df >> ply.slice_rows(5) >> ply.call(".transpose")

hetionet_cbg_df = pd.read_csv(cbg_url, sep="\t") >> ply.call(
    ".astype", {"entrez_gene_id": int}
)
hetionet_cbg_df.head(2)

# +
query = """
SELECT "compound_cid" AS drugbank_id, "gene_cid" AS entrez_gene_id, count(*) AS n_sentences
FROM compound_gene
GROUP BY "compound_cid", "gene_cid";
"""

compound_gene_sentence_df = (
    pd.read_sql(query, database_str)
    >> ply_tdy.separate_rows("entrez_gene_id", sep=";")
    >> ply.call(".astype", {"entrez_gene_id": int})
)
compound_gene_sentence_df.head(2)
# -

# ## Merge all dataframes into One

compound_binds_gene_df = (
    compound_gene_sentence_df
    >> ply.inner_join(
        drugbank_df >> ply.select("drugbank_id", "drug_name"), on="drugbank_id"
    )
    >> ply.inner_join(
        entrez_gene_df >> ply.select("entrez_gene_id", "gene_symbol"),
        on="entrez_gene_id",
    )
    >> ply.left_join(
        hetionet_cbg_df >> ply.select("drugbank_id", "entrez_gene_id", "sources"),
        on=["drugbank_id", "entrez_gene_id"],
    )
    >> ply.call(".fillna", {"n_sentences": 0})
    >> ply.call(".astype", {"n_sentences": int})
    >> ply.define(
        hetionet="sources.notnull().astype(int)",
        has_sentence="(n_sentences > 0).astype(int)",
    )
)
compound_binds_gene_df >> ply.slice_rows(5)

outfile = "compound_binds_gene.tsv.xz"
if not Path(outfile).exists():
    (
        compound_binds_gene_df
        >> ply.call(".to_csv", outfile, sep="\t", index=False, compression="xz")
    )

# free memory for rest of notebook
del drugbank_df
del entrez_gene_df
del compound_gene_sentence_df
del hetionet_cbg_df
del compound_binds_gene_df

# # Gene Interacts Gene

# ## Gene Info URLs for Hetionet

gene_url = "https://raw.githubusercontent.com/dhimmel/entrez-gene/a7362748a34211e5df6f2d185bb3246279760546/data/genes-human.tsv"
ppi_url = "https://raw.githubusercontent.com/dhimmel/ppi/f6a7edbc8de6ba2d7fe1ef3fee4d89e5b8d0b900/data/ppi-hetio-ind.tsv"

# ## Load DataFrames for each URL

entrez_gene_df = (
    pd.read_csv(gene_url, sep="\t")
    >> ply.rename(dict(entrez_gene_id="GeneID", gene_symbol="Symbol"))
    >> ply.define(merge_key=1)
)
entrez_gene_df >> ply.slice_rows(5)

hetionet_gig_df = pd.read_csv(ppi_url, sep="\t") >> ply.rename(
    dict(gene1_id="gene_0", gene2_id="gene_1")
)
hetionet_gig_df >> ply.slice_rows(5)

# +
query = """
SELECT "gene1_cid" AS gene1_id, "gene2_cid" AS gene2_id, count(*) AS n_sentences
FROM gene_gene
GROUP BY "gene1_cid", "gene2_cid";
"""

gene_gene_sentence_df = (
    pd.read_sql(query, database_str)
    >> ply_tdy.separate_rows("gene1_id", sep=";")
    >> ply_tdy.separate_rows("gene2_id", sep=";")
    >> ply.call(".astype", {"gene1_id": int, "gene2_id": int})
)
gene_gene_sentence_df >> ply.slice_rows(5)
# -

# ## Merge all dataframes into One

gene_interacts_gene_df = (
    gene_gene_sentence_df
    >> ply.left_join(hetionet_gig_df, on=["gene1_id", "gene2_id"])
    >> ply.call(".fillna", {"n_sentences": 0})
    >> ply.call(".astype", {"n_sentences": int})
    >> ply.define(
        hetionet="sources.notnull().astype(int)",
        has_sentence="(n_sentences > 0).astype(int)",
    )
    >> ply.inner_join(
        entrez_gene_df
        >> ply.select("entrez_gene_id", "gene_symbol")
        >> ply.rename(dict(gene1_id="entrez_gene_id", gene1_name="Symbol")),
        on="gene1_id",
    )
    >> ply.inner_join(
        entrez_gene_df
        >> ply.select("entrez_gene_id", "gene_symbol")
        >> ply.rename(dict(gene2_id="entrez_gene_id", gene2_name="Symbol")),
        on="gene2_id",
    )
)
gene_interacts_gene_df >> ply.slice_rows(5)

outfile = "gene_interacts_gene.tsv.xz"
if not Path(outfile).exists():
    (
        gene_interacts_gene_df
        >> ply.call(".to_csv", outfile, sep="\t", index=False, compression="xz")
    )

# free memory for rest of notebook
del entrez_gene_df
del gene_gene_sentence_df
del hetionet_gig_df
del gene_interacts_gene_df
