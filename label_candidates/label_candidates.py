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

# # Label the Candidate Sentences

# +
# %load_ext autoreload
# %autoreload 2

import os
import sys

from sqlalchemy import create_engine
from snorkel.labeling import PandasLFApplier
import pandas as pd

from snorkeling_helper.label_functions.disease_gene import DaG
# -

username = "danich1"
password = "snorkel"
dbname = "pubmed_central_db"
database_str = (
    f"postgresql+psycopg2://{username}:{password}@/{dbname}?host=/var/run/postgresql"
)
conn = create_engine(database_str)

candidates_df = pd.read_csv(
    "../annotation_conversion/output/mapped_disease_gene_candidates.tsv", sep="\t"
)
candidates_df.head()

# ## Train Candidates Abstract Only

# +
candidates = ",".join(map(str, candidates_df.document_id.tolist()))
sql = f"""
select * from disease_gene
where sentence_id not in ({candidates}) and
section = 'title' or section ='abstract'
"""

candidate = pd.read_sql(sql, database_str)
print(candidate.shape)
candidate.head()

# +
sen_id_map = ",".join(map(str, candidate.sentence_id.tolist()))
sql = f"""
select sentence.document_id, sentence_id
from sentence
inner join
(
    select distinct document_id from sentence
    where section != 'title' and section !='abstract'
) as doc_ids
on doc_ids.document_id=sentence.document_id
where sentence_id in ({sen_id_map})
"""

full_text_docs_ids = pd.read_sql(sql, database_str)
print(full_text_docs_ids.shape)
full_text_docs_ids.head()
# -

candidate = (
    candidate.merge(
        full_text_docs_ids.assign(full_text=True)[["sentence_id", "full_text"]],
        on=["sentence_id"],
        how="left",
    )
    .fillna(False)
    .assign(
        word=lambda x: x.apply(
            lambda row: row.word.replace("'", "").split("|"), axis=1
        ),
        pos_tag=lambda x: x.apply(
            lambda row: row.pos_tag.replace("'", "").split("|"), axis=1
        ),
    )
)
print(candidate.shape)
candidate.head()

# +
lf_columns = list(DaG["distant_supervision"].values()) + list(
    DaG["text_patterns"].values()
)

lf_column_names = [col.name for col in lf_columns]
# -

lf_applier = PandasLFApplier(lfs=lf_columns)
labels = lf_applier.apply(df=candidate)

train_lfs_df = pd.DataFrame(labels, columns=lf_column_names).assign(
    candidate_id=candidate.candidate_id.tolist()
)
print(train_lfs_df.shape)
train_lfs_df.head()

(train_lfs_df.to_csv("output/dg_abstract_train_candidates.tsv", sep="\t", index=False))

# ## Train Full Text Only

# +
candidates = ",".join(map(str, candidates_df.document_id.tolist()))
sql = f"""
select * from disease_gene
where sentence_id not in ({candidates}) and
section != 'title' and section !='abstract'
"""

candidate = pd.read_sql(sql, database_str)
print(candidate.shape)
candidate.head()
# -

candidate = candidate.assign(
    word=lambda x: x.apply(lambda row: row.word.replace("'", "").split("|"), axis=1),
    pos_tag=lambda x: x.apply(
        lambda row: row.pos_tag.replace("'", "").split("|"), axis=1
    ),
)
print(candidate.shape)
candidate.head()

# +
lf_columns = list(DaG["distant_supervision"].values()) + list(
    DaG["text_patterns"].values()
)

lf_column_names = [col.name for col in lf_columns]
# -

lf_applier = PandasLFApplier(lfs=lf_columns)
labels = lf_applier.apply(df=candidate)

train_lfs_df = pd.DataFrame(labels, columns=lf_column_names).assign(
    candidate_id=candidate.candidate_id.tolist()
)
print(train_lfs_df.shape)
train_lfs_df.head()

(train_lfs_df.to_csv("output/dg_full_text_train_candidates.tsv", sep="\t", index=False))

# ## Dev Set

annotated_df = pd.read_csv(
    "../annotation_conversion/output/mapped_disease_gene_candidates.tsv", sep="\t"
)
annotated_df.head()


def char_to_word(row):
    char_word_mapper = [
        (idx, int(tok)) for idx, tok in enumerate(row["char_offsets"].split("|"))
    ]
    gene_end = -1
    gene_start = -1
    disease_end = -1
    disease_start = -1
    for word in char_word_mapper:
        if word[1] >= row["disease_start"] and word[1] <= row["disease_end"]:
            if word[1] == row["disease_start"]:
                disease_start = word[0]
            else:
                disease_end = word[0]
        if word[1] >= row["gene_start"] and word[1] <= row["gene_end"]:
            if word[1] == row["gene_start"]:
                gene_start = word[0]
            else:
                gene_end = word[0]

    return (
        disease_start,
        disease_end if disease_end != -1 else disease_start + 1,
        gene_start,
        gene_end if gene_end != -1 else gene_start + 1,
    )


fixed_map_df = pd.DataFrame.from_dict(
    dict(
        zip(
            ["disease_start", "disease_end", "gene_start", "gene_end"],
            zip(*annotated_df.apply(char_to_word, axis=1).tolist()),
        )
    )
)
fixed_map_df

fixed_annotated_df = (
    annotated_df.assign(
        disease_start=fixed_map_df.disease_start.tolist(),
        disease_end=fixed_map_df.disease_end.tolist(),
        gene_start=fixed_map_df.gene_start.tolist(),
        gene_end=fixed_map_df.gene_end.tolist(),
    )
    .rename(
        index=str,
        columns={
            "Disease_cid": "disease_cid",
            "Gene_cid": "gene_cid",
            "words": "word",
            "lemmas": "lemma",
            "pos_tags": "pos_tag",
        },
    )
    .assign(
        lemma=lambda x: x.lemma.apply(lambda s: s.split("|")),
        word=lambda x: x.word.apply(lambda s: s.split("|")),
        pos_tag=lambda x: x.pos_tag.apply(lambda s: s.split("|")),
    )
)
fixed_annotated_df

# +
lf_columns = list(DaG["distant_supervision"].values()) + list(
    DaG["text_patterns"].values()
)

lf_column_names = [col.name for col in lf_columns]
# -

lf_applier = PandasLFApplier(lfs=lf_columns)
labels = lf_applier.apply(df=fixed_annotated_df)

dev_lfs_df = (
    pd.DataFrame(labels, columns=lf_column_names)
    # .assign(candidate_id=fixed_annotated_df.candidate_id.tolist())
)
print(dev_lfs_df.shape)
dev_lfs_df.head()

(
    dev_lfs_df.assign(
        split=fixed_annotated_df.split.tolist(),
        document_id=fixed_annotated_df.document_id.tolist(),
        curated_dsh=fixed_annotated_df.curated_dsh.tolist(),
    ).to_csv("output/dg_dev_test_candidates.tsv", sep="\t", index=False)
)
