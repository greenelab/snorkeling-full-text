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

# # Top Ten Predicted Sentences for each Edge Type

# +
# %load_ext autoreload
# %autoreload 2
# %matplotlib inline

from pathlib import Path

import pandas as pd
import plydata as ply
from snorkel.labeling.model import LabelModel
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

gen_performance_list = list(
    Path("../generative_model_training/output/performance").rglob("*performance.tsv")
)
gen_training_data_list = list(
    Path("../generative_model_training/output").rglob("*temp_abstract_file.tsv")
)
training_candidate_label_matrix = list(
    Path("../label_candidates/output").rglob(
        "*abstract_train_candidates_resampling.tsv"
    )
)
predicted_sentences_list = list(
    Path("../edge_prediction_experiment/output").rglob("*sentences.tsv")
)

gen_model_best_params = dict()
for dataframe_file in gen_performance_list:
    edge_label = dataframe_file.stem.split("_")[0]
    performance_df = pd.read_csv(dataframe_file, sep="\t")
    best_params = (
        performance_df
        >> ply.query("data_source=='abstract'")
        >> ply.query(f"label_source=='{edge_label}'")
        >> ply.query("lf_num==lf_num.max()")
        >> ply.query("model=='tune'")
        >> ply.select("epochs", "l2_param", "lr_param")
        >> ply.distinct()
    ).values[0]
    gen_model_best_params[edge_label.lower()] = list(best_params)

candidate_label_matrices = dict()
for label_matrix_datafile in training_candidate_label_matrix:
    edge_label = label_matrix_datafile.stem.split("_")[0]
    edge_label = (
        "dag"
        if edge_label == "dg"
        else "ctd"
        if edge_label == "cd"
        else "cbg"
        if edge_label == "cg"
        else "gig"
    )
    candidate_label_matrices[edge_label] = label_matrix_datafile

gen_model_training_data = dict()
for training_datafile in gen_training_data_list:
    edge_label = training_datafile.stem.split("_")[0]
    gen_model_training_data[edge_label] = training_datafile

lf_feature_list_map = dict(
    dag=dict(database=(0, 5), edge_specific_lfs=(5, 34)),
    ctd=dict(database=(0, 3), edge_specific_lfs=(32, 52)),
    cbg=dict(database=(0, 9), edge_specific_lfs=(60, 80)),
    gig=dict(database=(0, 9), edge_specific_lfs=(80, 108)),
)

edge_to_table_map = dict(
    dag=[
        "disease_gene",
        "disease_cid",
        "doid_id",
        "gene_cid",
        "entrez_gene_id",
        "disease_start",
        "disease_end",
        "gene_start",
        "gene_end",
    ],
    gig=[
        "gene_gene",
        "gene1_cid",
        "gene1_id",
        "gene2_cid",
        "gene2_id",
        "gene1_start",
        "gene1_end",
        "gene2_start",
        "gene2_end",
    ],
    cbg=[
        "compound_gene",
        "compound_cid",
        "drugbank_id",
        "gene_cid",
        "entrez_gene_id",
        "compound_start",
        "compound_end",
        "gene_start",
        "gene_end",
    ],
    ctd=[
        "compound_disease",
        "compound_cid",
        "drugbank_id",
        "disease_cid",
        "doid_id",
        "compound_start",
        "compound_end",
        "disease_start",
        "disease_end",
    ],
)

for predicted_edge_type in predicted_sentences_list:

    # Edge types
    edge_label = predicted_edge_type.stem.split("_")[2]

    if Path(f"output/table_one_{edge_label}.tsv").exists():
        continue

    # Load the sentences
    predicted_sentences_df = pd.read_csv(str(predicted_edge_type), sep="\t")
    predicted_sentences_df = (
        predicted_sentences_df >> ply.arrange("-pred") >> ply.slice_rows(40)
    )
    candidate_ids = predicted_sentences_df.candidate_id.tolist()

    # Load the training data
    training_data = pd.read_csv(
        gen_model_training_data[edge_label], sep="\t"
    ) >> ply.call(".drop", "candidate_id", axis=1)

    database_start = lf_feature_list_map[edge_label]["database"][0]
    database_end = lf_feature_list_map[edge_label]["database"][1]
    database_cols = list(training_data.columns[database_start:database_end])

    lf_start = lf_feature_list_map[edge_label]["edge_specific_lfs"][0]
    lf_end = lf_feature_list_map[edge_label]["edge_specific_lfs"][1]
    lf_cols = list(training_data.columns[lf_start:lf_end])
    total_features = database_cols + lf_cols

    prediction_data = (
        pd.read_csv(candidate_label_matrices[edge_label], sep="\t")
        >> ply.query(f"candidate_id in {candidate_ids}")
        >> ply.call(".drop_duplicates")
        >> ply.arrange("candidate_id")
    )

    # Train Generative Model to get predictions
    label_model = LabelModel(cardinality=2)
    label_model.fit(
        training_data[total_features].values,
        n_epochs=int(gen_model_best_params[edge_label][0]),
        l2=gen_model_best_params[edge_label][1],
        lr=gen_model_best_params[edge_label][2],
        log_freq=10,
        seed=100,
    )

    gen_model_pred = label_model.predict_proba(
        prediction_data
        >> ply.call(".drop", "candidate_id", axis=1)
        >> ply.select(*total_features)
    )[:, 1]

    mapped_fields = edge_to_table_map[edge_label]

    if len(candidate_ids) != prediction_data.candidate_id.shape[0]:
        candidate_id_list = ",".join(map(str, prediction_data.candidate_id.tolist()))
    else:
        candidate_id_list = ",".join(map(str, candidate_ids))

    # Get the sentences
    sql = f"""
     select
        candidate.candidate_id,
        {mapped_fields[1]} as {mapped_fields[2]},
        {mapped_fields[3]} as {mapped_fields[4]},
        lemma,
        entity_one_word_start as {mapped_fields[5]},
        entity_one_word_end as {mapped_fields[6]},
        entity_two_word_start as {mapped_fields[7]},
        entity_two_word_end as {mapped_fields[8]}
    from
        {mapped_fields[0]} inner join candidate
        on {mapped_fields[0]}.candidate_id=candidate.candidate_id
        where candidate.candidate_id in ({candidate_id_list})
    """

    candidate_to_sentence_map_df = (
        pd.read_sql(sql, conn)
        >> ply.arrange("candidate_id")
        >> ply.define(
            generative_model_pred=gen_model_pred,
            discriminative_model_pred=(
                predicted_sentences_df
                >> ply.query(
                    f"candidate_id in {list(map(int, candidate_id_list.split(',')))}"
                )
                >> ply.arrange("candidate_id")
                >> ply.pull("pred")
            ),
        )
    )

    candidate_to_sentence_map_df >> ply.call(
        ".to_csv", f"output/table_one_{edge_label}.tsv", sep="\t", index=False
    )

generated_table_files = list(Path("output").rglob("table_one_*tsv"))
node_map_files = Path("../snorkeling_helper/label_functions/knowledge_bases")
edge_file_map = dict(
    dag="disease_associates_gene.tsv.xz",
    ctd="compound_treats_disease.tsv.xz",
    cbg="compound_binds_gene.tsv.xz",
    gig="gene_interacts_gene.tsv.xz",
)


def identify_entities(entity_df, entity_one_type, entity_two_type):
    return_lemma_list = []
    for row_id, df_row in entity_df.iterrows():
        lemma_list = df_row["lemma"].replace("'", "").split("|")
        lemma_list[df_row[f"{entity_one_type}_start"]] = (
            "[" + lemma_list[df_row[f"{entity_one_type}_start"]]
        )
        lemma_list[df_row[f"{entity_one_type}_end"]] = (
            "].{"
            + f"{entity_one_type}_color"
            + "} "
            + lemma_list[df_row[f"{entity_one_type}_end"]]
        )

        lemma_list[df_row[f"{entity_two_type}_start"]] = (
            "[" + lemma_list[df_row[f"{entity_two_type}_start"]]
        )
        lemma_list[df_row[f"{entity_two_type}_end"]] = (
            "].{"
            + f"{entity_two_type}_color"
            + "} "
            + lemma_list[df_row[f"{entity_two_type}_end"]]
        )

        return_lemma_list.append(" ".join(lemma_list))
    return return_lemma_list


def finalize_table(
    results_table,
    entity_ids_to_concepts,
    entity_one_type="disease",
    entity_one_id_label="doid_id",
    entity_one_name="doid_name",
    entity_two_type="gene",
    entity_two_id_label="entrez_gene_id",
    entity_two_name="gene_symbol",
    edge_label="[D]{.disease_color}a[G]{.gene_color}",
):
    entity_pairs = " OR ".join(
        results_table
        >> ply.select(entity_one_id_label, entity_two_id_label)
        >> ply.distinct()
        >> ply.call(
            ".apply",
            lambda y: f"{entity_one_type}_cid='{y[entity_one_id_label]}' and {entity_two_type}_cid='{y[entity_two_id_label]}'",
            axis=1,
        )
    )

    first_entity = "".join(i for i in entity_one_type if not i.isdigit())
    second_entity = "".join(i for i in entity_two_type if not i.isdigit())

    sql = f"""
    select
        {entity_one_type}_cid as {entity_one_id_label},
        {entity_two_type}_cid as {entity_two_id_label},
        count(*) as n_sentences
    from (
        select * from {first_entity}_{second_entity}
        where {entity_pairs}
    ) as entity_match
    group by {entity_one_type}_cid, {entity_two_type}_cid;
    """

    if "gene" in entity_one_type and "gene" in entity_two_type:
        column_conversion = {
            f"{entity_one_id_label}": int,
            f"{entity_two_id_label}": int,
        }
        sentences_count_df = pd.read_sql(sql, conn) >> ply.call(
            ".astype", column_conversion, errors="ignore"
        )

    elif "gene" in entity_one_type:
        column_conversion = {f"{entity_one_id_label}": int}
        sentences_count_df = pd.read_sql(sql, conn) >> ply.call(
            ".astype", column_conversion, errors="ignore"
        )

    elif "gene" in entity_two_type:
        column_conversion = {f"{entity_two_id_label}": int}
        sentences_count_df = pd.read_sql(sql, conn) >> ply.call(
            ".astype", column_conversion, errors="ignore"
        )

    else:
        sentences_count_df = pd.read_sql(sql, conn)

    results_table = (
        results_table
        >> ply.define(
            lemma=lambda df: identify_entities(df, entity_one_type, entity_two_type)
        )
        >> ply.inner_join(
            sentences_count_df, on=[entity_one_id_label, entity_two_id_label]
        )
        >> ply.left_join(
            entity_ids_to_concepts
            >> ply.select(
                entity_one_id_label,
                entity_one_name,
                entity_two_id_label,
                entity_two_name,
                "hetionet",
            ),
            on=[entity_one_id_label, entity_two_id_label],
        )
        >> ply.distinct()
        >> ply.define(
            edge_type=f'"{edge_label}"',
            hetionet=ply.if_else("hetionet==0", '"Novel"', '"Existing"'),
        )
        >> ply.rename(source_node=entity_one_name, target_node=entity_two_name)
        >> ply.select(
            "edge_type",
            "source_node",
            "target_node",
            "generative_model_pred",
            "discriminative_model_pred",
            "n_sentences",
            "hetionet",
            "lemma",
            "candidate_id",
        )
    )

    return results_table


finalized_table = []
for sentence_table in generated_table_files:
    edge_label = sentence_table.stem.split("_")[2]
    results_table = pd.read_csv(sentence_table, sep="\t")
    entity_ids_to_concepts = pd.read_csv(
        str(node_map_files / edge_file_map[edge_label]), sep="\t"
    )

    if edge_label == "dag":
        results_table = finalize_table(
            results_table,
            entity_ids_to_concepts,
            entity_one_type="disease",
            entity_one_id_label="doid_id",
            entity_one_name="doid_name",
            entity_two_type="gene",
            entity_two_id_label="entrez_gene_id",
            entity_two_name="gene_symbol",
            edge_label="[D]{.disease_color}a[G]{.gene_color}",
        )

    if edge_label == "ctd":
        results_table = finalize_table(
            results_table,
            entity_ids_to_concepts,
            entity_one_type="compound",
            entity_one_id_label="drugbank_id",
            entity_one_name="drug_name",
            entity_two_type="disease",
            entity_two_id_label="doid_id",
            entity_two_name="doid_name",
            edge_label="[C]{.compound_color}t[D]{.disease_color}",
        )

    if edge_label == "cbg":
        results_table = finalize_table(
            results_table,
            entity_ids_to_concepts,
            entity_one_type="compound",
            entity_one_id_label="drugbank_id",
            entity_one_name="name",
            entity_two_type="gene",
            entity_two_id_label="entrez_gene_id",
            entity_two_name="gene_symbol",
            edge_label="[C]{.compound_color}b[G]{.gene_color}",
        )

    if edge_label == "gig":
        results_table = finalize_table(
            results_table,
            entity_ids_to_concepts,
            entity_one_type="gene1",
            entity_one_id_label="gene1_id",
            entity_one_name="gene1_name",
            entity_two_type="gene2",
            entity_two_id_label="gene2_id",
            entity_two_name="gene2_name",
            edge_label="[G]{.gene_color}i[G]{.gene_color}",
        )

    finalized_table.append(results_table)
final_table_df = pd.concat(finalized_table)
final_table_df

final_table_df >> ply.call(".to_csv", "output/table_one.tsv", sep="\t", index=False)

(
    final_table_df
    >> ply.select("-candidate_id")
    >> ply.group_by("edge_type")
    >> ply.arrange("-discriminative_model_pred")
    >> ply.ungroup()
    >> ply.call(
        ".to_csv",
        "output/table_one_formatted.tsv",
        sep="\t",
        index=False,
        float_format="%.3f",
    )
)
