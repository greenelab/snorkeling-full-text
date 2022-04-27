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

# # Output Parsed Lemmas for Disc Model

# This notebook is designed to output sentences into forms that BioBERT can process for training.

# +
import warnings

import pandas as pd
import plydata as ply
import spacy
from sqlalchemy import create_engine

from snorkeling_helper.candidates_helper import char_to_word, encode_lemmas

warnings.filterwarnings("ignore")
# -

username = "danich1"
password = "snorkel"
dbname = "pubmed_central_db"
database_str = (
    f"postgresql+psycopg2://{username}:{password}@/{dbname}?host=/var/run/postgresql"
)
conn = create_engine(database_str)

nlp = spacy.load("en_core_web_sm")
stopwords = nlp.Defaults.stop_words

# # DaG

# ## Dev-Test

hand_labeled_candidates_df = pd.read_csv(
    "../annotation_conversion/output" "/mapped_disease_gene_candidates.tsv", sep="\t"
)
hand_labeled_candidates_df.head()

fieldnames = ["disease_start", "disease_end", "gene_start", "gene_end"]

candidate_index_df = pd.DataFrame.from_dict(
    dict(
        zip(
            fieldnames,
            zip(
                *hand_labeled_candidates_df.apply(
                    char_to_word, row_fields=fieldnames, axis=1
                ).tolist()
            ),
        )
    )
)
candidate_index_df.head()

hand_labeled_candidates_df = (
    hand_labeled_candidates_df
    >> ply.rename(candidate_id="old_candidate_id", lemma="lemmas")
    >> ply.define(
        disease_start=candidate_index_df >> ply.pull("disease_start"),
        disease_end=candidate_index_df >> ply.pull("disease_end"),
        gene_start=candidate_index_df >> ply.pull("gene_start"),
        gene_end=candidate_index_df >> ply.pull("gene_end"),
        lemma=lambda x: x.lemma.apply(lambda y: y.split("|")),
    )
)
hand_labeled_candidates_df.head()

# +
encoded_dev_test = pd.DataFrame.from_records(
    encode_lemmas(
        hand_labeled_candidates_df,
        stopwords,
        dict(),
        entity_fieldnames=fieldnames,
        entity_one="DISEASE_ENTITY",
        entity_two="GENE_ENTITY",
    )
)

print(encoded_dev_test.shape)
encoded_dev_test.head()
# -

(
    encoded_dev_test
    >> ply.inner_join(
        hand_labeled_candidates_df
        >> ply.select("curated_dsh", "split", "candidate_id")
        >> ply.query("split>0"),
        on="candidate_id",
    )
    >> ply.call(
        ".to_csv", "output/dg_dev_test_encoded_lemmas.tsv", sep="\t", index=False
    )
)

# ## Train

dag_mapper = pd.read_csv(
    "../generative_model_training/output/dag_dataset_mapper.tsv", sep="\t"
)
dag_mapper.head()

train_docs = ",".join(
    map(str, dag_mapper.query("dataset == 'train'").document_id.tolist())
)
candidates = ",".join(map(str, hand_labeled_candidates_df.document_id.tolist()))

# +
sql = f"""
select candidate_id, document_id, text, lemma, disease_start, disease_end, gene_start, gene_end
from disease_gene inner join (
    select sentence_id, document_id
    from sentence
) as sentence_map
on disease_gene.sentence_id = sentence_map.sentence_id
where document_id in ({train_docs}) and
candidate_id not in ({candidates}) and
section = 'title' or section ='abstract'
"""

candidate = pd.read_sql(sql, database_str) >> ply.define(
    lemma=lambda x: x.lemma.apply(lambda y: y.replace("'", "").split("|"))
)
print(candidate.shape)
candidate.head()

# +
encoded_abstracts = pd.DataFrame.from_records(
    encode_lemmas(
        candidate,
        stopwords,
        dict(),
        entity_fieldnames=fieldnames,
        entity_one="DISEASE_ENTITY",
        entity_two="GENE_ENTITY",
    )
)

print(encoded_abstracts.shape)
encoded_abstracts.head()
# -

(
    encoded_abstracts
    >> ply.call(
        "to_csv" "output/train_dg_abstract_encoded_lemmas.tsv", sep="\t", index=False
    )
)

# # CtD

# ## Dev-Test

hand_labeled_candidates_df = pd.read_csv(
    "../annotation_conversion/output" "/mapped_compound_disease_candidates.tsv",
    sep="\t",
)
hand_labeled_candidates_df.head()

fieldnames = [
    "compound_start",
    "compound_end",
    "disease_start",
    "disease_end",
]

candidate_index_df = pd.DataFrame.from_dict(
    dict(
        zip(
            fieldnames,
            zip(
                *hand_labeled_candidates_df.apply(
                    char_to_word, row_fields=fieldnames, axis=1
                ).tolist()
            ),
        )
    )
)
candidate_index_df.head()

hand_labeled_candidates_df = (
    hand_labeled_candidates_df
    >> ply.rename(candidate_id="old_candidate_id", lemma="lemmas")
    >> ply.define(
        compound_start=candidate_index_df >> ply.pull("compound_start"),
        compound_end=candidate_index_df >> ply.pull("compound_end"),
        disease_start=candidate_index_df >> ply.pull("disease_start"),
        disease_end=candidate_index_df >> ply.pull("disease_end"),
        lemma=lambda x: x.lemma.apply(lambda y: y.split("|")),
    )
)
hand_labeled_candidates_df.head()

# +
encoded_dev_test = pd.DataFrame.from_records(
    encode_lemmas(
        hand_labeled_candidates_df,
        stopwords,
        dict(),
        entity_fieldnames=fieldnames,
        entity_one="COMPOUND_ENTITY",
        entity_two="DISEASE_ENTITY",
    )
)

print(encoded_dev_test.shape)
encoded_dev_test.head()
# -

(
    encoded_dev_test
    >> ply.inner_join(
        hand_labeled_candidates_df
        >> ply.select("curated_ctd", "split", "candidate_id"),
        on="candidate_id",
    )
    >> ply.call(
        ".to_csv", "output/cd_dev_test_encoded_lemmas.tsv", sep="\t", index=False
    )
)

# ## Train

ctd_mapper = pd.read_csv(
    "../generative_model_training/output/ctd_dataset_mapper.tsv", sep="\t"
)
ctd_mapper.head()

train_docs = ",".join(
    map(str, ctd_mapper.query("dataset == 'train'").document_id.tolist())
)
candidates = ",".join(map(str, hand_labeled_candidates_df.document_id.tolist()))

# +
sql = f"""
select candidate_id, document_id, text, lemma, compound_start, compound_end, disease_start, disease_end
from compound_disease inner join (
    select sentence_id, document_id
    from sentence
) as sentence_map
on compound_disease.sentence_id = sentence_map.sentence_id
where document_id in ({train_docs}) and
candidate_id not in ({candidates}) and
section = 'title' or section ='abstract'
"""

candidate = pd.read_sql(sql, database_str) >> ply.define(
    lemma=lambda x: x.lemma.apply(lambda y: y.replace("'", "").split("|"))
)
print(candidate.shape)
candidate.head()

# +
encoded_abstracts = pd.DataFrame.from_records(
    encode_lemmas(
        candidate,
        stopwords,
        dict(),
        entity_fieldnames=fieldnames,
        entity_one="COMPOUND_ENTITY",
        entity_two="DISEASE_ENTITY",
    )
)

print(encoded_abstracts.shape)
encoded_abstracts.head()
# -

(
    encoded_abstracts
    >> ply.call(
        "to_csv", "output/train_cd_abstract_encoded_lemmas.tsv", sep="\t", index=False
    )
)

# # CbG

# ## Dev-Test

hand_labeled_candidates_df = pd.read_csv(
    "../annotation_conversion/output" "/mapped_compound_gene_candidates.tsv", sep="\t"
)
hand_labeled_candidates_df.head()

fieldnames = ["compound_start", "compound_end", "gene_start", "gene_end"]

candidate_index_df = pd.DataFrame.from_dict(
    dict(
        zip(
            fieldnames,
            zip(
                *hand_labeled_candidates_df.apply(
                    char_to_word, row_fields=fieldnames, axis=1
                ).tolist()
            ),
        )
    )
)
candidate_index_df.head()

hand_labeled_candidates_df = (
    hand_labeled_candidates_df
    >> ply.rename(candidate_id="old_candidate_id", lemma="lemmas")
    >> ply.define(
        compound_start=candidate_index_df >> ply.pull("compound_start"),
        compound_end=candidate_index_df >> ply.pull("compound_end"),
        gene_start=candidate_index_df >> ply.pull("gene_start"),
        gene_end=candidate_index_df >> ply.pull("gene_end"),
        lemma=lambda x: x.lemma.apply(lambda y: y.split("|")),
    )
)
hand_labeled_candidates_df.head()

# +
encoded_dev_test = pd.DataFrame.from_records(
    encode_lemmas(
        hand_labeled_candidates_df,
        stopwords,
        dict(),
        entity_fieldnames=fieldnames,
        entity_one="COMPOUND_ENTITY",
        entity_two="GENE_ENTITY",
    )
)

print(encoded_dev_test.shape)
encoded_dev_test.head()
# -

(
    encoded_dev_test
    >> ply.inner_join(
        hand_labeled_candidates_df
        >> ply.select("curated_cbg", "split", "candidate_id"),
        on="candidate_id",
    )
    >> ply.call(
        ".to_csv", "output/cg_dev_test_encoded_lemmas.tsv", sep="\t", index=False
    )
)

# ## Train

cbg_mapper = pd.read_csv(
    "../generative_model_training/output/cbg_dataset_mapper.tsv", sep="\t"
)
cbg_mapper.head()

train_docs = ",".join(
    map(str, cbg_mapper.query("dataset == 'train'").document_id.tolist())
)
candidates = ",".join(map(str, hand_labeled_candidates_df.document_id.tolist()))

# +
sql = f"""
select candidate_id, document_id, text, lemma, compound_start, compound_end, gene_start, gene_end
from compound_gene inner join (
    select sentence_id, document_id
    from sentence
) as sentence_map
on compound_gene.sentence_id = sentence_map.sentence_id
where document_id in ({train_docs}) and
candidate_id not in ({candidates}) and
section = 'title' or section ='abstract'
"""

candidate = pd.read_sql(sql, database_str) >> ply.define(
    lemma=lambda x: x.lemma.apply(lambda y: y.replace("'", "").split("|"))
)
print(candidate.shape)
candidate.head()

# +
encoded_abstracts = pd.DataFrame.from_records(
    encode_lemmas(
        candidate,
        stopwords,
        dict(),
        entity_fieldnames=fieldnames,
        entity_one="COMPOUND_ENTITY",
        entity_two="GENE_ENTITY",
    )
)

print(encoded_abstracts.shape)
encoded_abstracts.head()
# -

(
    encoded_abstracts
    >> ply.call(
        "to_csv", "output/train_cg_abstract_encoded_lemmas.tsv", sep="\t", index=False
    )
)

# # GiG

# ## Dev-Test

hand_labeled_candidates_df = pd.read_csv(
    "../annotation_conversion/output" "/mapped_gene_gene_candidates.tsv", sep="\t"
)
hand_labeled_candidates_df.head()

fieldnames = ["gene1_start", "gene1_end", "gene2_start", "gene2_end"]

candidate_index_df = pd.DataFrame.from_dict(
    dict(
        zip(
            fieldnames,
            zip(
                *hand_labeled_candidates_df.apply(
                    char_to_word, row_fields=fieldnames, axis=1
                ).tolist()
            ),
        )
    )
)
candidate_index_df.head()

hand_labeled_candidates_df = (
    hand_labeled_candidates_df
    >> ply.rename(candidate_id="old_candidate_id", lemma="lemmas")
    >> ply.define(
        gene1_start=candidate_index_df >> ply.pull("gene1_start"),
        gene1_end=candidate_index_df >> ply.pull("gene1_end"),
        gene2_start=candidate_index_df >> ply.pull("gene2_start"),
        gene2_end=candidate_index_df >> ply.pull("gene2_end"),
        lemma=lambda x: x.lemma.apply(lambda y: y.split("|")),
    )
)
hand_labeled_candidates_df.head()

# +
encoded_dev_test = pd.DataFrame.from_records(
    encode_lemmas(
        hand_labeled_candidates_df,
        stopwords,
        dict(),
        entity_fieldnames=fieldnames,
        entity_one="GENE1_ENTITY",
        entity_two="GENE2_ENTITY",
    )
)

print(encoded_dev_test.shape)
encoded_dev_test.head()
# -

(
    encoded_dev_test
    >> ply.inner_join(
        hand_labeled_candidates_df
        >> ply.select("curated_gig", "split", "candidate_id"),
        on="candidate_id",
    )
    >> ply.call(
        ".to_csv", "output/gg_dev_test_encoded_lemmas.tsv", sep="\t", index=False
    )
)

# ## Train

gig_mapper = pd.read_csv(
    "../generative_model_training/output/gig_dataset_mapper.tsv", sep="\t"
)
gig_mapper.head()

train_docs = ",".join(
    map(str, gig_mapper.query("dataset == 'train'").document_id.tolist())
)
candidates = ",".join(map(str, hand_labeled_candidates_df.document_id.tolist()))

# +
sql = f"""
select candidate_id, document_id, text, lemma, gene1_start, gene1_end, gene2_start, gene2_end
from gene_gene inner join (
    select sentence_id, document_id
    from sentence
) as sentence_map
on gene_gene.sentence_id = sentence_map.sentence_id
where document_id in ({train_docs}) and
candidate_id not in ({candidates}) and
section = 'title' or section ='abstract'
"""

candidate = pd.read_sql(sql, database_str) >> ply.define(
    lemma=lambda x: x.lemma.apply(lambda y: y.replace("'", "").split("|"))
)
print(candidate.shape)
candidate.head()

# +
encoded_abstracts = pd.DataFrame.from_records(
    encode_lemmas(
        candidate,
        stopwords,
        dict(),
        entity_fieldnames=fieldnames,
        entity_one="GENE1_ENTITY",
        entity_two="GENE2_ENTITY",
    )
)

print(encoded_abstracts.shape)
encoded_abstracts.head()
# -

(
    encoded_abstracts
    >> ply.call(
        "to_csv", "output/train_gg_abstract_encoded_lemmas.tsv", sep="\t", index=False
    )
)
