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

# # Measure Gene interacts Gene Edge Recall

# This notebook is designed to estimate how many hetionet edges can be recalled from this weakly supervised approach.
# After the discriminator model has been trained, the last step is to predict whether every candidate sentence mentions a gene interacts gene relationship.
# Following the predictions, the next step is to group each candidate sentence based on each candidate pair and take the max value of each group.
# This max value represents the probability that an edge should or does exist.
# Lastly, once each candidate has been scored I use the precision recall curve to estimate how many edges can be recovered from this method.

# +
# %load_ext autoreload
# %autoreload 2
# %matplotlib inline

import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotnine as p9
import plydata as ply
import plydata.tidy as ply_tdy
from sklearn.metrics import precision_recall_curve, roc_curve, auc
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

hetionet_gig_map_df = pd.read_csv(
    Path("../snorkeling_helper/label_functions/knowledge_bases")
    / "gene_interacts_gene.tsv.xz",
    sep="\t",
)
hetionet_gig_map_df >> ply.slice_rows(5)

sentence_prediction_df = pd.read_csv(
    "output/all_predicted_gig_sentences.tsv", sep="\t"
) >> ply.arrange("candidate_id")
sentence_prediction_df >> ply.slice_rows(5)

sql = """
select candidate_id, gene1_cid as gene1_id, gene2_cid as gene2_id
from gene_gene inner join (
    select sentence_id, document_id
    from sentence
) as sentence_map
on gene_gene.sentence_id = sentence_map.sentence_id
where section = 'title' or section ='abstract'
"""
candidate_to_sentence_map_df = (
    pd.read_sql(sql, conn)
    >> ply_tdy.separate_rows("gene1_id", sep=";")
    >> ply_tdy.separate_rows("gene2_id", sep=";")
    >> ply.call(".astype", {"gene1_id": int, "gene2_id": int})
)
candidate_to_sentence_map_df >> ply.slice_rows(5)

# # Merge Predictions with Candidates and Hetionet Map

all_gig_predictions_df = (
    sentence_prediction_df
    >> ply.inner_join(candidate_to_sentence_map_df, on="candidate_id")
    >> ply.inner_join(
        hetionet_gig_map_df >> ply.select("-sources", "-n_sentences"),
        on=["gene1_id", "gene2_id"],
    )
)
all_gig_predictions_df >> ply.slice_rows(5)

all_gig_df = (
    all_gig_predictions_df
    >> ply.group_by("gene1_id", "gene2_id")
    >> ply.define(
        pred_max="max(pred)", pred_mean="mean(pred)", pred_median="median(pred)"
    )
    >> ply.select("-pred", "-candidate_id")
    >> ply.distinct()
    >> ply.ungroup()
    >> ply_tdy.gather("metric", "score", ["pred_max", "pred_mean", "pred_median"])
)
all_gig_df >> ply.slice_rows(10)

test_entity_df = all_gig_df >> ply.query("split == 5")
test_entity_df >> ply.slice_rows(5)

# # Determine Precision and Recall

performance_map = dict()

# +
precision, recall, pr_threshold = precision_recall_curve(
    test_entity_df >> ply.query("metric=='pred_max'") >> ply.pull("hetionet"),
    test_entity_df >> ply.query("metric=='pred_max'") >> ply.pull("score"),
)

fpr, tpr, roc_threshold = roc_curve(
    test_entity_df >> ply.query("metric=='pred_max'") >> ply.pull("hetionet"),
    test_entity_df >> ply.query("metric=='pred_max'") >> ply.pull("score"),
)

performance_map["PR"] = (
    pd.DataFrame(
        {
            "precision": precision,
            "recall": recall,
            "pr_threshold": np.append(pr_threshold, 1),
        }
    )
    >> ply.define(model=f'"pred_max/AUC={auc(recall, precision):.2f}"')
)

performance_map["AUROC"] = pd.DataFrame(
    {"fpr": fpr, "tpr": tpr, "roc_threshold": roc_threshold}
) >> ply.define(model=f'"pred_max/AUC={auc(fpr, tpr):.2f}"')

# +
precision, recall, pr_threshold = precision_recall_curve(
    test_entity_df >> ply.query("metric=='pred_mean'") >> ply.pull("hetionet"),
    test_entity_df >> ply.query("metric=='pred_mean'") >> ply.pull("score"),
)

fpr, tpr, roc_threshold = roc_curve(
    test_entity_df >> ply.query("metric=='pred_mean'") >> ply.pull("hetionet"),
    test_entity_df >> ply.query("metric=='pred_mean'") >> ply.pull("score"),
)

performance_map["PR"] = performance_map["PR"].append(
    pd.DataFrame(
        {
            "precision": precision,
            "recall": recall,
            "pr_threshold": np.append(pr_threshold, 1),
        }
    )
    >> ply.define(model=f'"pred_mean/AUC={auc(recall, precision):.2f}"')
)

performance_map["AUROC"] = performance_map["AUROC"].append(
    pd.DataFrame({"fpr": fpr, "tpr": tpr, "roc_threshold": roc_threshold})
    >> ply.define(model=f'"pred_mean/AUC={auc(fpr, tpr):.2f}"')
)

# +
precision, recall, pr_threshold = precision_recall_curve(
    test_entity_df >> ply.query("metric=='pred_median'") >> ply.pull("hetionet"),
    test_entity_df >> ply.query("metric=='pred_median'") >> ply.pull("score"),
)

fpr, tpr, roc_threshold = roc_curve(
    test_entity_df >> ply.query("metric=='pred_median'") >> ply.pull("hetionet"),
    test_entity_df >> ply.query("metric=='pred_median'") >> ply.pull("score"),
)

performance_map["PR"] = performance_map["PR"].append(
    pd.DataFrame(
        {
            "precision": precision,
            "recall": recall,
            "pr_threshold": np.append(pr_threshold, 1),
        }
    )
    >> ply.define(model=f'"pred_median/AUC={auc(recall, precision):.2f}"')
)

performance_map["AUROC"] = performance_map["AUROC"].append(
    pd.DataFrame({"fpr": fpr, "tpr": tpr, "roc_threshold": roc_threshold})
    >> ply.define(model=f'"pred_median/AUC={auc(fpr, tpr):.2f}"')
)
# -

g = (
    p9.ggplot(
        performance_map["AUROC"]
        >> ply.call(
            ".append",
            pd.DataFrame(
                {
                    "fpr": [0, 0.25, 0.5, 0.75, 1],
                    "tpr": [0, 0.25, 0.5, 0.75, 1],
                    "model": "random",
                }
            ),
        )
    )
    + p9.aes(x="fpr", y="tpr", group="model", color="model", linetype="model")
    + p9.geom_line()
    + p9.theme_seaborn("white")
    + p9.scale_color_manual(["#1b9e77", "#d95f02", "#7570b3", "#000000"])
    + p9.scale_linetype_manual(["solid", "solid", "solid", "dashed"])
)
print(g)

g = (
    p9.ggplot(performance_map["PR"])
    + p9.aes(x="recall", y="precision", group="model", color="model", linetype="model")
    + p9.geom_line()
    + p9.theme_seaborn("white")
    + p9.scale_color_manual(["#1b9e77", "#d95f02", "#7570b3", "#000000"])
    + p9.scale_linetype_manual(["solid", "solid", "solid", "dashed"])
    + p9.geom_hline(
        yintercept=(
            test_entity_df >> ply.query("metric=='pred_median'") >> ply.pull("hetionet")
        ).mean(),
        linetype="dashed",
    )
)
print(g)

# ## Estimate number of new Edges Added

# +
df_iterator = (
    performance_map["PR"]
    >> ply.query("model.str.contains('max')& pr_threshold < 1")
    >> ply.call(".round", {"pr_threshold": 2})
    >> ply.distinct("pr_threshold", "last")
)

edges_added_records = []
for idx, row in df_iterator.iterrows():
    cutoff = row["pr_threshold"]

    values_added = (
        all_gig_df
        >> ply.query("metric.str.contains('max')")
        >> ply.query("score >= @cutoff")
        >> ply.pull("hetionet")
    )

    edges_added_records.append(
        {
            "edges": values_added.sum(),
            "in_hetionet": "Existing",
            "precision": row["precision"],
            "sen_cutoff": cutoff,
        }
    )

    edges_added_records.append(
        {
            "edges": values_added.shape[0] - values_added.sum(),
            "in_hetionet": "Novel",
            "precision": row["precision"],
            "sen_cutoff": cutoff,
        }
    )


edges_added_df = pd.DataFrame.from_records(edges_added_records)
edges_added_df >> ply.slice_rows(10)
# -

g = (
    p9.ggplot(edges_added_df >> ply.query("edges > 0 & precision > 0"))
    + p9.aes(x="precision", y="edges", color="in_hetionet")
    + p9.geom_point()
    + p9.theme_seaborn("white")
    + p9.scale_color_brewer(type="qual", palette=2)
    + p9.scale_y_log10()
)
print(g)

cutoff_score = roc_threshold[np.argmax(tpr - fpr)]
print(cutoff_score)

# +
edges_df = pd.DataFrame.from_records(
    [
        {
            "recall": (
                all_gig_df
                >> ply.query("metric=='pred_max' & score > @cutoff_score")
                >> ply.pull("hetionet")
            ).sum()
            / all_gig_df.query("hetionet == 1").shape[0],
            "edges": (
                all_gig_df
                >> ply.query("metric=='pred_max' & score > @cutoff_score")
                >> ply.pull("hetionet")
            ).sum(),
            "in_hetionet": "Existing",
            "relation": "GiG",
        },
        {
            "edges": (
                all_gig_df
                >> ply.query("metric=='pred_max' & score > @cutoff_score")
                >> ply.query("hetionet==0")
            ).shape[0],
            "in_hetionet": "Novel",
            "relation": "GiG",
        },
    ]
)

edges_df >> ply.call(".to_csv", "output/GiG_edge_recall.tsv", sep="\t", index=False)
edges_df
# -

g = (
    p9.ggplot(edges_df, p9.aes(x="relation", y="edges", fill="in_hetionet"))
    + p9.geom_col(position="dodge")
    + p9.geom_text(
        p9.aes(
            label=(
                edges_df.apply(
                    lambda x: f"{x['edges']} ({x['recall']*100:.0f}%)"
                    if not math.isnan(x["recall"])
                    else f"{x['edges']}",
                    axis=1,
                )
            )
        ),
        position=p9.position_dodge(width=1),
        size=9,
        va="bottom",
    )
    + p9.scale_y_log10()
    + p9.theme(
        axis_text_y=p9.element_blank(),
        axis_ticks_major=p9.element_blank(),
        rect=p9.element_blank(),
    )
)
print(g)

# # Take home messages

# 1. Recall is okay. Achieves 29% which is better than originally thought.
# 2. If the discriminator model trained correctly I hypothesize that the recall would be a lot more higher.
