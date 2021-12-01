# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
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

# # Resort Candidates Based on Document ID

# +
import warnings

import numpy as np
import pandas as pd
import plydata as ply
from sqlalchemy import create_engine

warnings.filterwarnings("ignore")
# -

username = "danich1"
password = "snorkel"
dbname = "pubmed_central_db"
database_str = (
    f"postgresql+psycopg2://{username}:{password}@/{dbname}?host=/var/run/postgresql"
)
conn = create_engine(database_str)


def resample_candidates(sql: str, L_dev: pd.DataFrame) -> pd.DataFrame:
    candidate_doc_df = pd.read_sql(sql, database_str)
    filtered_candidate_id = (
        candidate_doc_df
        >> ply.query(f"document_id in {list(L_dev.document_id.astype(int).unique())}")
        >> ply.pull("candidate_id")
    )

    np.random.seed(100)
    sorted_train_df = (
        candidate_doc_df
        >> ply.query(f"candidate_id not in {filtered_candidate_id}")
        >> ply.select("document_id")
        >> ply.distinct()
        >> ply.define(
            dataset=lambda x: np.random.choice(
                ["train", "tune", "test"], x.shape[0], p=[0.7, 0.2, 0.1]
            )
        )
    )

    return sorted_train_df


# ## DaG

# Grab the document ids for resampling
sql = """
select dg_candidates.sentence_id, document_id, dg_candidates.candidate_id from sentence
inner join (
  select candidate.candidate_id, disease_gene.sentence_id from disease_gene
  inner join candidate on candidate.candidate_id=disease_gene.candidate_id
  ) as dg_candidates
on sentence.sentence_id = dg_candidates.sentence_id
"""
candidate_doc_df = pd.read_sql(sql, database_str)
candidate_doc_df.head()

L_dev = pd.read_csv(
    "../label_candidates/output/dg_dev_test_candidates_resampling.tsv", sep="\t"
) >> ply.query("split==1")
print(L_dev.shape)
L_dev.head().T

sorted_train_df = resample_candidates(sql, L_dev)
sorted_train_df.to_csv("output/DaG/dag_dataset_mapper.tsv", sep="\t", index=False)
sorted_train_df.head()

# ## CtD

# Grab the document ids for resampling
sql = """
select cd_candidates.sentence_id, document_id, cd_candidates.candidate_id
from sentence
inner join (
  select candidate.candidate_id, compound_disease.sentence_id from compound_disease
  inner join candidate on candidate.candidate_id=compound_disease.candidate_id
  ) as cd_candidates
on sentence.sentence_id = cd_candidates.sentence_id
"""

L_dev = pd.read_csv(
    "../label_candidates/output/cd_dev_test_candidates_resampling.tsv", sep="\t"
) >> ply.query("split==10")
print(L_dev.shape)
L_dev.head().T

sorted_train_df = resample_candidates(sql, L_dev)
sorted_train_df.to_csv("output/CtD/ctd_dataset_mapper.tsv", sep="\t", index=False)
sorted_train_df.head()

# ## CbG

# Grab the document ids for resampling
sql = """
select cg_candidates.sentence_id, document_id, cg_candidates.candidate_id from sentence
inner join (
  select candidate.candidate_id, compound_gene.sentence_id from compound_gene
  inner join candidate on candidate.candidate_id=compound_gene.candidate_id
  ) as cg_candidates
on sentence.sentence_id = cg_candidates.sentence_id
"""

L_dev = pd.read_csv(
    "../label_candidates/output/cg_dev_test_candidates_resampling.tsv", sep="\t"
) >> ply.query("split==7")
print(L_dev.shape)
L_dev.head().T

sorted_train_df = resample_candidates(sql, L_dev)
sorted_train_df.to_csv("output/CbG/cbg_dataset_mapper.tsv", sep="\t", index=False)
sorted_train_df.head()

# ## GiG

L_dev = pd.read_csv(
    "../label_candidates/output/gg_dev_test_candidates_resampling.tsv", sep="\t"
).query("split==4")
print(L_dev.shape)
L_dev.head().T

# Grab the document ids for resampling
sql = """
select gg_candidates.sentence_id, document_id, gg_candidates.candidate_id from sentence
inner join (
  select candidate.candidate_id, gene_gene.sentence_id from gene_gene
  inner join candidate on candidate.candidate_id=gene_gene.candidate_id
  ) as gg_candidates
on sentence.sentence_id = gg_candidates.sentence_id
"""

sorted_train_df = resample_candidates(sql, L_dev)
sorted_train_df.to_csv("output/GiG/gig_dataset_mapper.tsv", sep="\t", index=False)
sorted_train_df.head()
