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

# # Does resampling experiment help with predicting GiG sentences?

# +
from itertools import product
from pathlib import Path
import warnings

import pandas as pd
import plydata as ply
from sqlalchemy import create_engine

from snorkel.labeling.analysis import LFAnalysis
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

label_candidates_dir = Path("../label_candidates/output")
notebook_output_dir = Path("output/GiG")

# +
L_abstracts = pd.read_csv(
    str(label_candidates_dir / Path("gg_abstract_train_candidates_resampling.tsv")),
    sep="\t",
)

print(L_abstracts.shape)
L_abstracts.head().T
# -

L_dev = pd.read_csv(
    str(label_candidates_dir / Path("gg_dev_test_candidates_resampling.tsv")), sep="\t"
) >> ply.query("split==4")
print(L_dev.shape)
L_dev.head().T

# ## Resort Based on the Candidate Abstracts

# Grab the document ids for resampling
sql = """
select gg_candidates.sentence_id, document_id, gg_candidates.candidate_id from sentence
inner join (
  select candidate.candidate_id, gene_gene.sentence_id from gene_gene
  inner join candidate on candidate.candidate_id=gene_gene.candidate_id
  ) as gg_candidates
on sentence.sentence_id = gg_candidates.sentence_id
"""
candidate_doc_df = pd.read_sql(sql, database_str)
candidate_doc_df.head()

# +
dev_test_ids = (
    L_dev >> ply.select("document_id") >> ply.distinct() >> ply.pull("document_id")
)

filtered_candidate_id = (
    candidate_doc_df
    >> ply.query(f"document_id in {list(dev_test_ids)}")
    >> ply.pull("candidate_id")
)
# -

sorted_train_df = pd.read_csv(
    str(notebook_output_dir / Path("gig_dataset_mapper.tsv")), sep="\t"
)
sorted_train_df.head()

# ## Load full text after document sorting

trained_documents = (
    sorted_train_df
    >> ply.inner_join(candidate_doc_df, on="document_id")
    >> ply.query("dataset=='train'")
    >> ply.pull("candidate_id")
)

# ## Update the data based on sorting

filtered_L_abstracts = L_abstracts >> ply.query(
    f"candidate_id in {list(trained_documents)}"
)
print(filtered_L_abstracts.shape)
filtered_L_abstracts.head()

# ## Construct the Grid Search

# Global Grid
epochs_grid = [100]
l2_param_grid = [0.75]
lr_grid = [1e-3]
grid = list(product(epochs_grid, l2_param_grid, lr_grid))

# # Abstracts

# +
analysis_module = LFAnalysis(
    filtered_L_abstracts >> ply.select("candidate_id", drop=True)
)

abstract_lf_summary = analysis_module.lf_summary()
abstract_lf_summary.index = (
    filtered_L_abstracts >> ply.select("candidate_id", drop=True)
).columns.tolist()

abstract_lf_summary
# -

# # Set up fields for resampling

lf_columns_base = list(L_abstracts.columns[0:9])
candidate_id_field = list(L_abstracts.columns[-1:])
dev_column_base = ["split", "curated_gig", "document_id"]
data_columns = []

# # Abstracts

# ## Baseline

# +
gig_start = 0
gig_end = 9
number_of_samples = 1

gig_lf_range = range(gig_start, gig_end)
size_of_samples = [len(gig_lf_range)]
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
    marginals_df_file=str(
        notebook_output_dir / Path("gig_training_marginals_baseline.tsv")
    ),
    curated_label="curated_gig",
    entity_label="GiG",
    data_source="abstract",
)

# ## DaG

# +
dag_start = 9
dag_end = 38

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
    marginals_df_file=str(
        notebook_output_dir / Path("dag_predicts_gig_training_marginals.tsv")
    ),
    curated_label="curated_gig",
    entity_label="DaG",
    data_source="abstract",
)

# ## CtD

# +
ctd_start = 38
ctd_end = 60

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
    marginals_df_file=str(
        notebook_output_dir / Path("ctd_predicts_gig_training_marginals.tsv")
    ),
    curated_label="curated_gig",
    entity_label="CtD",
    data_source="abstract",
)

# ## CbG

# +
cbg_start = 60
cbg_end = 80

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
    marginals_df_file=str(
        notebook_output_dir / Path("cbg_predicts_gig_training_marginals.tsv")
    ),
    curated_label="curated_gig",
    entity_label="CbG",
    data_source="abstract",
)

# ## GiG

# +
gig_start = 80
gig_end = 108

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
    marginals_df_file=str(
        notebook_output_dir / Path("gig_predicts_gig_training_marginals.tsv")
    ),
    curated_label="curated_gig",
    entity_label="GiG",
    data_source="abstract",
)

# # Full Text

# Full text cannot load into memory on my work desktop machine (RAM:64GB).
# Would have to run it on a cluster that has more memory; however, given the fact that full text hasn't improved performance I'm electing to ignore this section and work with abstracts alone.
# Check [06_plot_labels_sampling_performance.ipynb](06_plot_labels_sampling_performance.ipynb) for full text and abstract analysis results.

# # Write Performance to File

performance_df = pd.DataFrame.from_records(data_columns)
performance_df

performance_df.to_csv("output/performance/GiG_performance.tsv", index=False, sep="\t")
