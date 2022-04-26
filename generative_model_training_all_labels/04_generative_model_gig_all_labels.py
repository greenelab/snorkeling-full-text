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
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import plydata as ply
from sqlalchemy import create_engine

from snorkel.labeling.analysis import LFAnalysis
from snorkeling_helper.generative_model_helper import (
    sample_lfs,
    run_generative_label_function_sampler,
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
notebook_output_dir = Path("../generative_model_training/output/GiG")

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

L_test = pd.read_csv(
    str(label_candidates_dir / Path("gg_dev_test_candidates_resampling.tsv")), sep="\t"
) >> ply.query("split==5")
print(L_test.shape)
L_test.head().T

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
epochs_grid = [500]
l2_param_grid = np.linspace(0.01, 5, num=5)
lr_grid = [1e-2]
grid = list(
    zip(epochs_grid * len(l2_param_grid), l2_param_grid, lr_grid * len(l2_param_grid))
)

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

# ## GiG

# +
gig_start = 9
gig_end = 108

# Spaced out number of sampels including total
size_of_samples = [1, 33, 65, 97, gig_end - gig_start]
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

data_columns += run_generative_label_function_sampler(
    filtered_L_abstracts,
    L_dev,
    L_test,
    sampled_lfs_dict,
    lf_columns_base=lf_columns_base,
    grid_param=grid,
    marginals_df_file="",
    curated_label="curated_gig",
    entity_label="ALL",
    data_source="abstract",
)

# # Write Performance to File

performance_df = pd.DataFrame.from_records(data_columns)
performance_df

(
    performance_df
    >> ply.call(
        "to_csv",
        str(Path("output") / Path("ALL_GiG_performance.tsv")),
        index=False,
        sep="\t",
    )
)