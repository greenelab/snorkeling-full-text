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

# # Extract All Sentences for Final Evaluation

# This notebook is designed to output all sentences for BioBERT so that we can predict the existence of edge types downstream.

# +
import warnings

import pandas as pd
import plydata as ply
import spacy
from sqlalchemy import create_engine

from snorkeling_helper.candidates_helper import encode_lemmas

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

# +
sql = """
select candidate_id, document_id, text, lemma, disease_start, disease_end, gene_start, gene_end
from disease_gene inner join (
    select sentence_id, document_id
    from sentence
) as sentence_map
on disease_gene.sentence_id = sentence_map.sentence_id
where section = 'title' or section ='abstract'
"""

candidate = pd.read_sql(sql, database_str) >> ply.define(
    lemma=lambda x: x.lemma.apply(lambda y: y.replace("'", "").split("|"))
)
print(candidate.shape)
candidate.head()
# -

fieldnames = ["disease_start", "disease_end", "gene_start", "gene_end"]

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
        "to_csv", "output/all_dg_abstract_encoded_lemmas.tsv", sep="\t", index=False
    )
)

# # CtD

# +
sql = """
select candidate_id, document_id, text, lemma, compound_start, compound_end, disease_start, disease_end
from compound_disease inner join (
    select sentence_id, document_id
    from sentence
) as sentence_map
on compound_disease.sentence_id = sentence_map.sentence_id
where section = 'title' or section ='abstract'
"""

candidate = pd.read_sql(sql, database_str) >> ply.define(
    lemma=lambda x: x.lemma.apply(lambda y: y.replace("'", "").split("|"))
)
print(candidate.shape)
candidate.head()
# -

fieldnames = [
    "compound_start",
    "compound_end",
    "disease_start",
    "disease_end",
]

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
        "to_csv", "output/all_cd_abstract_encoded_lemmas.tsv", sep="\t", index=False
    )
)

# # CbG

# +
sql = """
select candidate_id, document_id, text, lemma, compound_start, compound_end, gene_start, gene_end
from compound_gene inner join (
    select sentence_id, document_id
    from sentence
) as sentence_map
on compound_gene.sentence_id = sentence_map.sentence_id
where section = 'title' or section ='abstract'
"""

candidate = pd.read_sql(sql, database_str) >> ply.define(
    lemma=lambda x: x.lemma.apply(lambda y: y.replace("'", "").split("|"))
)
print(candidate.shape)
candidate.head()
# -

fieldnames = [
    "compound_start",
    "compound_end",
    "gene_start",
    "gene_end",
]

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
        "to_csv", "output/all_cg_abstract_encoded_lemmas.tsv", sep="\t", index=False
    )
)

# # GiG

# +
sql = """
select candidate_id, document_id, text, lemma, gene1_start, gene1_end, gene2_start, gene2_end
from gene_gene inner join (
    select sentence_id, document_id
    from sentence
) as sentence_map
on gene_gene.sentence_id = sentence_map.sentence_id
where section = 'title' or section ='abstract'
"""

candidate = pd.read_sql(sql, database_str) >> ply.define(
    lemma=lambda x: x.lemma.apply(lambda y: y.replace("'", "").split("|"))
)
print(candidate.shape)
candidate.head()
# -

fieldnames = ["gene1_start", "gene1_end", "gene2_start", "gene2_end"]

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
        "to_csv", "output/all_gg_abstract_encoded_lemmas.tsv", sep="\t", index=False
    )
)
