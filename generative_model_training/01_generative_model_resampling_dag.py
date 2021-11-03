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

# # Does resampling experiment help with predicting DaG sentences?

# +
from itertools import product
from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
import plotnine as p9
import scipy.stats
from sqlalchemy import create_engine
from snorkel.labeling.analysis import LFAnalysis
from snorkel.labeling.model import LabelModel
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import torch
import torch.nn.functional as F
import tqdm

from snorkeling_helper.generative_model_helper import (
    sample_lfs,
    train_generative_label_function_sampler,
)

warnings.filterwarnings("ignore")
# -

username = "danich1"
password = "snorkel"
dbname = "pubmed_central_db"
database_str = (
    f"postgresql+psycopg2://{username}:{password}@/{dbname}?host=/var/run/postgresql"
)
conn = create_engine(database_str)

# ## Load the data

# +
L_abstracts = pd.read_csv(
    "../label_candidates/output/dg_abstract_train_candidates_resampling.tsv", sep="\t"
)

print(L_abstracts.shape)
L_abstracts.head().T

# +
L_full_text = pd.read_csv(
    "../label_candidates/output/dg_full_text_train_candidates_resampling.tsv", sep="\t"
)

print(L_full_text.shape)
L_full_text.head().T
# -

L_dev = pd.read_csv(
    "../label_candidates/output/dg_dev_test_candidates_resampling.tsv", sep="\t"
).query("split==1")
print(L_dev.shape)
L_dev.head().T

# ## Resort the Candidates Based on Abstract

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

filtered_candidate_id = candidate_doc_df.query(
    f"document_id in {list(L_dev.document_id.astype(int).unique())}"
).candidate_id.tolist()

if not Path("output/dag_dataset_mapper.tsv").exists():
    np.random.seed(100)
    sorted_train_df = (
        candidate_doc_df.query(
            f"document_id not in {list(L_dev.document_id.astype(int).unique())}"
        )[["document_id"]]
        .drop_duplicates()
        .assign(
            dataset=lambda x: np.random.choice(
                ["train", "tune", "test"], x.shape[0], p=[0.7, 0.2, 0.1]
            )
        )
    )
    sorted_train_df.to_csv("output/dag_dataset_mapper.tsv", sep="\t", index=False)
else:
    sorted_train_df = pd.read_csv("output/dag_dataset_mapper.tsv", sep="\t")
sorted_train_df.head()

trained_documents = (
    sorted_train_df.merge(candidate_doc_df, on="document_id")
    .query("dataset=='train'")
    .candidate_id.tolist()
)

filtered_L_abstracts = L_abstracts.query(f"candidate_id in {trained_documents}")
print(filtered_L_abstracts.shape)
filtered_L_abstracts.head()

filtered_L_full_text = L_full_text.query(f"candidate_id in {trained_documents}")
print(filtered_L_full_text.shape)
filtered_L_full_text.head()

# ## Construct the Grid Search

# Global Grid
epochs_grid = [100]
l2_param_grid = [0.75]
lr_grid = [1e-3]
grid = list(product(epochs_grid, l2_param_grid, lr_grid))

# # Abstracts

# +
analysis_module = LFAnalysis(filtered_L_abstracts.drop(["candidate_id"], axis=1))

abstract_lf_summary = analysis_module.lf_summary()
abstract_lf_summary.index = filtered_L_abstracts.drop(
    ["candidate_id"], axis=1
).columns.tolist()
abstract_lf_summary
# -

# # Set up For Resampling

lf_columns_base = list(L_abstracts.columns[0:5])
candidate_id_field = list(L_abstracts.columns[-1:])
dev_column_base = ["split", "curated_dsh", "document_id"]
data_columns = []

# # Abstracts

# ## Baseline

# +
dag_start = 0
dag_end = 5
number_of_samples = 1

dag_lf_range = range(dag_start, dag_end)
size_of_samples = [len(dag_lf_range)]
# -

sampled_lfs_dict = {
    sample_size: (
        sample_lfs(
            list(dag_lf_range),
            len(list(dag_lf_range)),
            sample_size,
            number_of_samples,
            random_state=100,
        )
    )
    for sample_size in size_of_samples
}

data_columns += train_generative_label_function_sampler(
    filtered_L_abstracts,
    L_dev,
    sampled_lfs_dict,
    lf_columns_base=lf_columns_base,
    candidate_id_field=candidate_id_field,
    dev_column_base=dev_column_base,
    search_grid=grid,
    marginals_df_file="output/DaG/dag_training_marginals_baseline.tsv",
    curated_label="curated_dsh",
    entity_label="DaG",
    data_source="abstract",
)

# ## DaG

# +
dag_start = 5
dag_end = 34

# Spaced out number of sampels including total
size_of_samples = [1, 6, 11, 16, dag_end - dag_start]
number_of_samples = 50
dag_lf_range = range(dag_start, dag_end)
# -

sampled_lfs_dict = {
    sample_size: (
        sample_lfs(
            list(dag_lf_range),
            len(list(dag_lf_range)),
            sample_size,
            number_of_samples,
            random_state=100,
        )
    )
    for sample_size in size_of_samples
}

data_columns += train_generative_label_function_sampler(
    filtered_L_abstracts,
    L_dev,
    sampled_lfs_dict,
    lf_columns_base=lf_columns_base,
    candidate_id_field=candidate_id_field,
    dev_column_base=dev_column_base,
    search_grid=grid,
    marginals_df_file="output/DaG/dag_predicts_dag_training_marginals.tsv",
    curated_label="curated_dsh",
    entity_label="DaG",
    data_source="abstract",
)

# ## CtD

# +
ctd_start = 34
ctd_end = 56

# Spaced out number of sampels including total
size_of_samples = [1, 6, 11, 16, ctd_end - ctd_start]
number_of_samples = 50
ctd_lf_range = range(ctd_start, ctd_end)
# -

sampled_lfs_dict = {
    sample_size: (
        sample_lfs(
            list(ctd_lf_range),
            len(list(ctd_lf_range)),
            sample_size,
            number_of_samples,
            random_state=100,
        )
    )
    for sample_size in size_of_samples
}

data_columns += train_generative_label_function_sampler(
    filtered_L_abstracts,
    L_dev,
    sampled_lfs_dict,
    lf_columns_base=lf_columns_base,
    candidate_id_field=candidate_id_field,
    dev_column_base=dev_column_base,
    search_grid=grid,
    marginals_df_file="output/DaG/ctd_predicts_dag_training_marginals.tsv",
    curated_label="curated_dsh",
    entity_label="CtD",
    data_source="abstract",
)

# ## CbG

# +
cbg_start = 56
cbg_end = 76

# Spaced out number of sampels including total
size_of_samples = [1, 6, 11, 16, cbg_end - cbg_start]
number_of_samples = 50
cbg_lf_range = range(cbg_start, cbg_end)
# -

sampled_lfs_dict = {
    sample_size: (
        sample_lfs(
            list(cbg_lf_range),
            len(list(cbg_lf_range)),
            sample_size,
            number_of_samples,
            random_state=100,
        )
    )
    for sample_size in size_of_samples
}

data_columns += train_generative_label_function_sampler(
    filtered_L_abstracts,
    L_dev,
    sampled_lfs_dict,
    lf_columns_base=lf_columns_base,
    candidate_id_field=candidate_id_field,
    dev_column_base=dev_column_base,
    search_grid=grid,
    marginals_df_file="output/DaG/cbg_predicts_dag_training_marginals.tsv",
    curated_label="curated_dsh",
    entity_label="CbG",
    data_source="abstract",
)

# ## GiG

# +
gig_start = 76
gig_end = 104

# Spaced out number of sampels including total
size_of_samples = [1, 6, 11, 16, gig_end - gig_start]
number_of_samples = 50
gig_lf_range = range(gig_start, gig_end)
# -

sampled_lfs_dict = {
    sample_size: (
        sample_lfs(
            list(gig_lf_range),
            len(list(gig_lf_range)),
            sample_size,
            number_of_samples,
            random_state=100,
        )
    )
    for sample_size in size_of_samples
}

data_columns += train_generative_label_function_sampler(
    filtered_L_abstracts,
    L_dev,
    sampled_lfs_dict,
    lf_columns_base=lf_columns_base,
    candidate_id_field=candidate_id_field,
    dev_column_base=dev_column_base,
    search_grid=grid,
    marginals_df_file="output/DaG/gig_predicts_dag_training_marginals.tsv",
    curated_label="curated_dsh",
    entity_label="GiG",
    data_source="abstract",
)

# # Full Text

# ## DaG

# +
dag_start = 5
dag_end = 24

# Spaced out number of sampels including total
size_of_samples = [1, 6, 11, 16, dag_end - dag_start]
number_of_samples = 50
dag_lf_range = range(dag_start, dag_end)
# -

sampled_lfs_dict = {
    sample_size: (
        sample_lfs(
            list(dag_lf_range),
            len(list(dag_lf_range)),
            sample_size,
            number_of_samples,
            random_state=100,
        )
    )
    for sample_size in size_of_samples
}

data_columns += train_generative_label_function_sampler(
    L_full_text,
    L_dev,
    sampled_lfs_dict,
    lf_columns_base=lf_columns_base,
    candidate_id_field=candidate_id_field,
    dev_column_base=dev_column_base,
    search_grid=grid,
    marginals_df_file="output/DaG/dag_predicts_dag_training_marginals_full_text.tsv",
    curated_label="curated_dsh",
    entity_label="DaG",
    data_source="full_text",
)

# ## CtD

# +
ctd_start = 34
ctd_end = 56

# Spaced out number of sampels including total
size_of_samples = [1, 6, 11, 16, ctd_end - ctd_start]
number_of_samples = 50
ctd_lf_range = range(ctd_start, ctd_end)
# -

sampled_lfs_dict = {
    sample_size: (
        sample_lfs(
            list(ctd_lf_range),
            len(list(ctd_lf_range)),
            sample_size,
            number_of_samples,
            random_state=100,
        )
    )
    for sample_size in size_of_samples
}

data_columns += train_generative_label_function_sampler(
    L_full_text,
    L_dev,
    sampled_lfs_dict,
    lf_columns_base=lf_columns_base,
    candidate_id_field=candidate_id_field,
    dev_column_base=dev_column_base,
    search_grid=grid,
    marginals_df_file="output/DaG/ctd_predicts_dag_training_marginals_full_text.tsv",
    curated_label="curated_dsh",
    entity_label="CtD",
    data_source="full_text",
)

# ## CbG

# +
cbg_start = 56
cbg_end = 76

# Spaced out number of sampels including total
size_of_samples = [1, 6, 11, 16, cbg_end - cbg_start]
number_of_samples = 50
cbg_lf_range = range(cbg_start, cbg_end)
# -

sampled_lfs_dict = {
    sample_size: (
        sample_lfs(
            list(cbg_lf_range),
            len(list(cbg_lf_range)),
            sample_size,
            number_of_samples,
            random_state=100,
        )
    )
    for sample_size in size_of_samples
}

data_columns += train_generative_label_function_sampler(
    L_full_text,
    L_dev,
    sampled_lfs_dict,
    lf_columns_base=lf_columns_base,
    candidate_id_field=candidate_id_field,
    dev_column_base=dev_column_base,
    search_grid=grid,
    marginals_df_file="output/DaG/cbg_predicts_dag_training_marginals_full_text.tsv",
    curated_label="curated_dsh",
    entity_label="CbG",
    data_source="full_text",
)

# ## GiG

# +
gig_start = 76
gig_end = 104

# Spaced out number of sampels including total
size_of_samples = [1, 6, 11, 16, gig_end - gig_start]
number_of_samples = 50
gig_lf_range = range(gig_start, gig_end)
# -

sampled_lfs_dict = {
    sample_size: (
        sample_lfs(
            list(gig_lf_range),
            len(list(gig_lf_range)),
            sample_size,
            number_of_samples,
            random_state=100,
        )
    )
    for sample_size in size_of_samples
}

data_columns += train_generative_label_function_sampler(
    L_full_text,
    L_dev,
    sampled_lfs_dict,
    lf_columns_base=lf_columns_base,
    candidate_id_field=candidate_id_field,
    dev_column_base=dev_column_base,
    search_grid=grid,
    marginals_df_file="output/DaG/gig_predicts_dag_training_marginals_full_text.tsv",
    curated_label="curated_dsh",
    entity_label="GiG",
    data_source="full_text",
)

# # Write Performance to File

performance_df = pd.DataFrame.from_records(data_columns)
performance_df

performance_df.to_csv("output/performance/DaG_performance.tsv", index=False, sep="\t")
