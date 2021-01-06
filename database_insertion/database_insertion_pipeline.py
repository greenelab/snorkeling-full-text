# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.9.1
#   kernelspec:
#     display_name: Python [conda env:snorkeling_full_text]
#     language: python
#     name: conda-env-snorkeling_full_text-py
# ---

# # Set up Import for Entire Pipeline

# +
## Default Librariess
from collections import Counter, OrderedDict, defaultdict
import csv
import itertools
import lzma
from pathlib import Path
import sys


## External Libraries
import dill as pickle
import intervaltree
from joblib import Parallel, delayed
import lxml.etree as ET
from multiprocessing import Process, Manager
import pandas as pd
from sqlalchemy import (
    create_engine,
    Integer,
    SmallInteger,
    BigInteger,
    MetaData,
    Table,
    String,
    VARCHAR,
    ARRAY,
    Column,
    Index,
    ForeignKey,
    UniqueConstraint,
)
import tqdm

## Custom module for pipeline
from snorkeling_helper.database_helper import (
    insert_entities,
    get_sql_result,
    parse_document,
    supply_documents,
)
# -

# # Run the Insertion pipeline

# This notebook is designed to import all tagged text from Pubtator Central into a Postgres database. This notebook can take at least a week to run, so be patient as large NLP projects can take a gargantuan amount of time to run.

# ## Create the Database

# This section invokes sqlalchemy's connection module to create a database called pubmed_central_db and populate the database with the following table declaration. The database name can be changed to whichever name you'd like.

# +
username = "####"  # replace with personal postgres username
password = "#####"  # replace with personal postgres password
dbname = "pubmed_central_db"
database_str = (
    f"postgresql+psycopg2://{username}:{password}@/{dbname}?host=/var/run/postgresql"
)
conn = create_engine(database_str)

metadata = MetaData()
sentence = Table(
    "sentence",
    metadata,
    Column("sentence_id", BigInteger, primary_key=True),
    Column("document_id", Integer),
    Column("section", String),
    Column("position", SmallInteger),
    Column("text", String),
    Column("word", String),
    Column("pos_tag", String),
    Column("lemma", String),
    Column("dep", String),
    Column("char_offset", String),
    Index("ix_sentence_document_id", "document_id"),
    UniqueConstraint("document_id", "section", "position", name="sentence_integrity"),
)

entity = Table(
    "entity",
    metadata,
    Column("entity_id", BigInteger, primary_key=True),
    Column("document_id", Integer),
    Column("entity_type", VARCHAR),
    Column("entity_cids", VARCHAR),
    Column("start", Integer),
    Column("end", Integer),
    Index("ix_entity_document_id", "document_id"),
    UniqueConstraint("document_id", "start", "end", name="entity_integrity"),
)

candidate = Table(
    "candidate",
    metadata,
    Column("candidate_id", BigInteger, primary_key=True),
    Column("candidate_type", VARCHAR),
    Column("dataset", SmallInteger),
    Column("entity_one_id", ForeignKey("entity.entity_id", ondelete="CASCADE")),
    Column("entity_two_id", ForeignKey("entity.entity_id", ondelete="CASCADE")),
    Column("entity_one_word_start", Integer),
    Column("entity_one_word_end", Integer),
    Column("entity_two_word_start", Integer),
    Column("entity_two_word_end", Integer),
    Column("sentence_id", ForeignKey("sentence.sentence_id", ondelete="CASCADE")),
)

metadata.create_all(conn)
# -

# ## Insert Entities

# This section is designed to populate the entities table via postgres copy command at the bottom of this notebook. This section first outputs all entities into a single file that will copied onto a postgres table. Make sure you have executed the [pubtator module](https://github.com/greenelab/pubtator) first before running this notebook.

Path("output").mkdir(exists_ok=True)  # create output folder if doesn't exist
already_seen = set()
seen_entity = set()
last_entity_id = 1

# +
pmcid_map_df = pd.read_csv(
    "../../pubtator/data/pubtator-pmids-to-pmcids.tsv", sep="\t"
)[["PMCID", "PMID"]]

already_seen, last_entity_id, _ = insert_entities(
    "../../pubtator/data/pubtator-central-full-hetnet-tags.tsv.xz",
    pmcid_map=dict(zip(pmcid_map_df.PMCID.values, pmcid_map_df.PMID.values)),
    already_seen=already_seen,
    entity_id_start=last_entity_id,
    output_file="output/pubmed_central_entities.tsv",
    seen_entity=seen_entity,
)

print(len(already_seen))

# +
already_seen, last_entity_id, _ = insert_entities(
    "../../pubtator/data/pubtator-central-hetnet-tags.tsv.xz",
    already_seen=already_seen,
    output_file="output/pubmed_central_entities.tsv",
    entity_id_start=last_entity_id,
    skip_documents=True,
)

print(len(already_seen))
# -

# Keep track of documents that have entities
(
    pd.DataFrame(list(already_seen), columns=["document_id"])
    .drop_duplicates()
    .to_csv("output/documents_with_entities.tsv", sep="\t", index=False)
)

# ## Insert Sentences

# This step takes the longest time and requires heavy computation. Uses Spacy to parse sentences and writes features to individual files. Each file will contain every sentence parsed by spacy. Make sure you have enough disk space to run this section of the notebook.

Path("output/files").mkdir(exists_ok=True)  # create output folder if doesn't exists
n_jobs = 3  # number of processes to run

# +
print("Starting Full Text Insertion")
full_path_map = {
    "document_id_path": "passage/infon[contains(@key, 'article-id_pmid')]/text()",
    "passage_path": "passage[infon[contains(@key, 'section_type')]]",
    "section_path": "infon[contains(@key, 'section_type')]/text()",
    "offset_path": "offset/text()",
    "section_text_path": "text/text()",
}

fieldnames = [
    "document_id",
    "section",
    "position",
    "text",
    "word",
    "pos_tag",
    "lemma",
    "dep",
    "char_offset",
]

with Manager() as m:
    data_queue = m.JoinableQueue(500000)
    jobs = []

    # Start the jobs
    for job in range(n_jobs):
        p = Process(target=parse_document, args=(full_path_map, fieldnames, data_queue))
        jobs.append(p)
        p.start()

    # Throw the documents onto the queue
    supply_documents(
        "full",
        Path("pubtator_central_batch").rglob("*xml"),  # Batch for full text extraction
        data_queue,
    )

    # Tell the jobs to end
    for job in range(n_jobs):
        data_queue.put(None)  # poison pill to end the processes

    for running_process in jobs:
        running_process.join()

# +
print("Starting Abstract Insertion")
abs_path_map = {
    "document_id_path": "id/text()",
    "passage_path": "passage[infon[contains(@key, 'type')]]",
    "section_path": "infon[contains(@key, 'type')]/text()",
    "offset_path": "offset/text()",
    "section_text_path": "text/text()",
}

fieldnames = [
    "document_id",
    "section",
    "position",
    "text",
    "word",
    "pos_tag",
    "lemma",
    "dep",
    "char_offset",
]

tag_generator = ET.iterparse(
    lzma.open(
        "../../pubtator/data/mar_1/pubtator-central-docs.xml.xz", "rb"
    ),  # abstract path
    tag="document",
    encoding="utf-8",
    recover=True,
)

with Manager() as m:
    data_queue = m.JoinableQueue(1000000)
    jobs = []
    # Start the jobs
    for job in range(n_jobs):
        p = Process(target=parse_document, args=(abs_path_map, fieldnames, data_queue))
        jobs.append(p)
        p.start()

    # Throw the documents onto the queue
    supply_documents("abstract", tag_generator, data_queue)

    # Tell the jobs to end
    for job in range(n_jobs):
        data_queue.put(None)  # poison pill to end the processes

    for running_process in jobs:
        running_process.join()
# -

# ## Merge Sentence files

# This section merges all the individual files into a centralized location. This makes it easier to quickly fill sentences into a postgres database (down below).

csv.field_size_limit(sys.maxsize)
with open("output/all_pubtator_central_docs.tsv", "w") as outfile:
    writer = csv.DictWriter(
        outfile,
        fieldnames=[
            "sentence_id",
            "document_id",
            "section",
            "position",
            "text",
            "word",
            "pos_tag",
            "lemma",
            "dep",
            "char_offset",
        ],
        delimiter="\t",
    )
    writer.writeheader()
    sentence_id = 1

    for doc_file in tqdm(Path("output/files").rglob("*tsv")):
        with open(doc_file, "r") as infile:
            reader = csv.DictReader(
                infile,
                fieldnames=[
                    "document_id",
                    "section",
                    "position",
                    "text",
                    "word",
                    "pos_tag",
                    "lemma",
                    "dep",
                    "char_offset",
                ],
                delimiter="\t",
            )

            for row in reader:
                try:
                    document_id = int(row["document_id"])
                    row["sentence_id"] = sentence_id
                    writer.writerow(row)
                    sentence_id += 1
                except Exception as e:
                    print(e)
                    print(row["document_id"])
                    print("Not a valid row skipping!!")
                    continue

# ## Insert Entities and Sentences into database

# This section inserts the data into a postgres database. Uses postgres copy function which is amazing at loading large data quickly.

# +
entity_sql = (
    f"copy public.entity from {Path('output/pubmed_central_entities.tsv').absolute()} "
    + "delimiter E'\t' csv HEADER;"
)
print(conn.execute(entity_sql))

entity_idx_sql = (
    "select setval('entity_entity_id_seq', (select count(*)+1 from entity), false);"
)
print(conn.execute(entity_idx_sql))

# +
sentence_sql = (
    f"copy public.sentence from {Path('output/all_pubtator_central_docs.tsv').absolute()} "
    + "delimiter E'\t' csv header;"
)
print(conn.execute(sentence_sql))

sentence_idx_sql = "select setval('sentence_sentence_id_seq', (select count(*)+1 from sentence), false);"
print(conn.execute(sentence_idx_sql))
# -

# ## Candidate Extraction

# This section extracts candidates from the loaded sentences above. After loading sentences from each document it uses an interval tree to find multiple entities in the same sentence. These sentences with multiple entities are considered candidates.

document_ids = get_sql_result(
    """
    SELECT DISTINCT document_id FROM entity;
    """,
    conn,
)

with open("output/candidates.tsv", "w") as outfile:

    # Create the writer
    writer = csv.DictWriter(
        outfile,
        fieldnames=[
            "candidate_id",
            "candidate_type",
            "dataset",
            "entity_one_id",
            "entity_two_id",
            "entity_one_word_start",
            "entity_one_word_end",
            "entity_two_word_start",
            "entity_two_word_end",
            "sentence_id",
        ],
        delimiter="\t",
    )
    writer.writeheader()
    candidate_id = 1

    for document in tqdm.tqdm(document_ids):
        entity_rows = get_sql_result(
            "SELECT * FROM entity " f"WHERE document_id = {document['document_id']}",
            conn,
        )
        sentence_rows = get_sql_result(
            "SELECT * FROM sentence " f"WHERE document_id = {document['document_id']}",
            conn,
        )

        entity_tree = intervaltree.IntervalTree.from_tuples(
            zip(
                list(map(lambda x: x["start"], entity_rows)),
                list(map(lambda x: x["end"], entity_rows)),
                list(
                    zip(
                        list(map(lambda x: x["entity_id"], entity_rows)),
                        list(map(lambda x: x["entity_type"], entity_rows)),
                    )
                ),
            )
        )

        for sentence in sentence_rows:
            char_list = list(map(int, sentence["char_offset"].split("|")))

            potential_candidates = entity_tree.overlap(
                begin=min(char_list), end=max(char_list)
            )

            if len(potential_candidates) < 2:
                continue

            word_offset = dict(zip(char_list, range(len(char_list))))

            for cand in itertools.combinations(potential_candidates, 2):
                entity_one_start = cand[0].begin
                entity_two_start = cand[1].begin

                if (
                    entity_one_start not in word_offset
                    or entity_two_start not in word_offset
                ):
                    continue

                entity_one_end = cand[0].end
                entity_two_end = cand[1].end

                if entity_one_end not in word_offset:
                    entity_one_end += 1

                    if entity_one_end not in word_offset:
                        continue

                if entity_two_end not in word_offset:
                    entity_two_end += 1

                    if entity_two_end not in word_offset:
                        continue

                writer.writerow(
                    {
                        "candidate_id": candidate_id,
                        "candidate_type": (
                            f"{cand[0].data[1].lower()}_{cand[1].data[1].lower()}"
                            if cand[0].data[1].lower() < cand[1].data[1].lower()
                            else f"{cand[1].data[1].lower()}_{cand[0].data[1].lower()}"
                        ),
                        "dataset": -1,
                        "entity_one_id": cand[0].data[0],
                        "entity_two_id": cand[1].data[0],
                        "entity_one_word_start": word_offset[entity_one_start],
                        "entity_one_word_end": word_offset[entity_one_end],
                        "entity_two_word_start": word_offset[entity_two_start],
                        "entity_two_word_end": word_offset[entity_two_end],
                        "sentence_id": sentence["sentence_id"],
                    }
                )

                candidate_id += 1

# +
candidate_sql = (
    f"copy public.candidate from {Path('output/candidates.tsv').absolute()} "
    + "delimiter E'\t' csv header;"
)
print(conn.execute(candidate_sql))

candidate_idx_sql = "select setval('candidate_candidate_id_seq', (select count(*)+1 from candidate), false);"
print(conn.execute(candidate_idx_sql))
# -

# ## Create Views

# There are lot of tables being constructed and filled, so to make your life easier I provided a database view that extracts the necessary information for each candidate. Once this is completed, it should be easy to work with candidates down the road.

disease_gene_view = """
CREATE VIEW disease_gene AS
SELECT
    CASE WHEN type[1] = 'Disease' THEN cids[1]  ELSE cids[2] end as disease_cid,
    CASE WHEN type[1] = 'Disease' THEN cids[2]  ELSE cids[1] end as gene_cid,
    candidate_id,
    f.sentence_id,
    sentence.section,
    sentence.text,
    sentence.word,
    sentence.pos_tag,
    sentence.char_offset,
    CASE WHEN type[1] = 'Disease' THEN f.entity_one_start ELSE f.entity_two_start end as disease_start,
    CASE WHEN type[1] = 'Disease' THEN f.entity_one_end  ELSE f.entity_two_end end as disease_end,
    CASE WHEN type[1] = 'Disease' THEN f.entity_two_start  ELSE f.entity_one_start end as gene_start,
    CASE WHEN type[1] = 'Disease' THEN f.entity_two_end  ELSE f.entity_one_end end as gene_end
    from (
         select
           array_agg(z.entity_cids) as cids,
           array_agg(z.entity_type) as type,
           min(candidate_id) as candidate_id,
           min(sentence_id) as sentence_id,
           min(z.entity_one_word_start) as entity_one_start,
           min(z.entity_one_word_end) as entity_one_end,
           min(z.entity_two_word_start) as entity_two_start,
           min(z.entity_two_word_end) as entity_two_end
         from (
                select
                  entity_cids,
                  candidate_id,
                  sentence_id,
                  candidate.entity_one_word_start,
                  candidate.entity_one_word_end,
                  candidate.entity_two_word_start,
                  candidate.entity_two_word_end,
                  entity_type
                from entity
                inner join candidate on entity.entity_id=candidate.entity_one_id or entity.entity_id=candidate.entity_two_id
                where candidate_type = 'disease_gene'
                     ) z
        group by candidate_id
        ) f
    inner join sentence on f.sentence_id=sentence.sentence_id;
"""
conn.execute(disease_gene_view)

compound_disease_view = """
CREATE VIEW compound_disease AS
SELECT
    CASE WHEN type[1] = 'Compound' THEN cids[1] ELSE cids[2] end as compound_cid,
    CASE WHEN type[1] = 'Compound' THEN cids[2] ELSE cids[1] end as disease_cid,
    candidate_id,
    f.sentence_id,
    sentence.section,
    sentence.text,
    sentence.word,
    sentence.pos_tag,
    sentence.char_offset,
    CASE WHEN type[1] = 'Compound' THEN f.entity_one_start ELSE f.entity_two_start end as compound_start,
    CASE WHEN type[1] = 'Compound' THEN f.entity_one_end  ELSE f.entity_two_end end as compound_end,
    CASE WHEN type[1] = 'Compound' THEN f.entity_two_start  ELSE f.entity_one_start end as disease_start,
    CASE WHEN type[1] = 'Compound' THEN f.entity_two_end  ELSE f.entity_one_end end as disease_end
    from (
         select
           array_agg(z.entity_cids) as cids,
           array_agg(z.entity_type) as type,
           min(candidate_id) as candidate_id,
           min(sentence_id) as sentence_id,
           min(z.entity_one_word_start) as entity_one_start,
           min(z.entity_one_word_end) as entity_one_end,
           min(z.entity_two_word_start) as entity_two_start,
           min(z.entity_two_word_end) as entity_two_end
         from (
                select
                  entity_cids,
                  candidate_id,
                  sentence_id,
                  candidate.entity_one_word_start,
                  candidate.entity_one_word_end,
                  candidate.entity_two_word_start,
                  candidate.entity_two_word_end,
                  entity_type
                from entity
                inner join candidate on entity.entity_id=candidate.entity_one_id or entity.entity_id=candidate.entity_two_id
                where candidate_type = 'compound_disease'
                     ) z
        group by candidate_id
        ) f
    inner join sentence on f.sentence_id=sentence.sentence_id;
"""
conn.execute(compound_disease_view)

compound_gene_view = """
CREATE VIEW compound_gene AS
SELECT
    CASE WHEN type[1] = 'Compound' THEN cids[1]  ELSE cids[2] end as compound_cid,
    CASE WHEN type[1] = 'Compound' THEN cids[2]  ELSE cids[1] end as gene_cid,
    candidate_id,
    f.sentence_id,
    sentence.section,
    sentence.text,
    sentence.word,
    sentence.pos_tag,
    sentence.char_offset,
    CASE WHEN type[1] = 'Compound' THEN f.entity_one_start ELSE f.entity_two_start end as compound_start,
    CASE WHEN type[1] = 'Compound' THEN f.entity_one_end  ELSE f.entity_two_end end as compound_end,
    CASE WHEN type[1] = 'Compound' THEN f.entity_two_start  ELSE f.entity_one_start end as gene_start,
    CASE WHEN type[1] = 'Compound' THEN f.entity_two_end  ELSE f.entity_one_end end as gene_end
    from (
         select
           array_agg(z.entity_cids) as cids,
           array_agg(z.entity_type) as type,
           min(candidate_id) as candidate_id,
           min(sentence_id) as sentence_id,
           min(z.entity_one_word_start) as entity_one_start,
           min(z.entity_one_word_end) as entity_one_end,
           min(z.entity_two_word_start) as entity_two_start,
           min(z.entity_two_word_end) as entity_two_end
         from (
                select
                  entity_cids,
                  candidate_id,
                  sentence_id,
                  candidate.entity_one_word_start,
                  candidate.entity_one_word_end,
                  candidate.entity_two_word_start,
                  candidate.entity_two_word_end,
                  entity_type
                from entity
                inner join candidate on entity.entity_id=candidate.entity_one_id or entity.entity_id=candidate.entity_two_id
                where candidate_type = 'compound_gene'
                     ) z
        group by candidate_id
        ) f
    inner join sentence on f.sentence_id=sentence.sentence_id;
"""
conn.execute(compound_gene_view)

gene_gene_view = """
CREATE VIEW gene_gene AS
SELECT
    CASE WHEN entity_one_start < entity_two_start THEN cids[1]  ELSE cids[2] end as gene1_cid,
    CASE WHEN entity_one_start < entity_two_start THEN cids[2]  ELSE cids[1] end as gene2_cid,
    candidate_id,
    f.sentence_id,
    sentence.section,
    sentence.text,
    sentence.word,
    sentence.pos_tag,
    sentence.char_offset,
    CASE WHEN entity_one_start < entity_two_start THEN f.entity_one_start ELSE f.entity_two_start end as gene1_start,
    CASE WHEN entity_one_start < entity_two_start THEN f.entity_one_end  ELSE f.entity_two_end end as gene1_end,
    CASE WHEN entity_one_start < entity_two_start THEN f.entity_two_start  ELSE f.entity_one_start end as gene2_start,
    CASE WHEN entity_one_start < entity_two_start THEN f.entity_two_end ELSE f.entity_one_end end as gene2_end
    from (
         select
           array_agg(z.entity_cids) as cids,
           array_agg(z.entity_type) as type,
           min(candidate_id) as candidate_id,
           min(sentence_id) as sentence_id,
           min(z.entity_one_word_start) as entity_one_start,
           min(z.entity_one_word_end) as entity_one_end,
           min(z.entity_two_word_start) as entity_two_start,
           min(z.entity_two_word_end) as entity_two_end
         from (
                select
                  entity_cids,
                  candidate_id,
                  sentence_id,
                  candidate.entity_one_word_start,
                  candidate.entity_one_word_end,
                  candidate.entity_two_word_start,
                  candidate.entity_two_word_end,
                  entity_type
                from entity
                inner join candidate on entity.entity_id=candidate.entity_one_id or entity.entity_id=candidate.entity_two_id
                where candidate_type = 'gene_gene'
                     ) z
        group by candidate_id
        ) f
    inner join sentence on f.sentence_id=sentence.sentence_id;
"""
conn.execute(gene_gene_view)
