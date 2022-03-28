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

# # Manuscript Figure Generation

# +
import math
from pathlib import Path

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import plotnine as p9
import plydata as ply
# -

# # Generative Model Figures

color_names = {
    "turquoise": np.array([27, 158, 119, 255]) / 255,
    "orange": np.array([217, 95, 2, 255]) / 255,
    "purple": np.array([117, 112, 179, 255]) / 255,
    "pink": np.array([231, 41, 138, 255]) / 255,
    "light-green": np.array([102, 166, 30, 255]) / 255,
}

color_map = {
    "DaG": mcolors.to_hex(color_names["turquoise"]),
    "CtD": mcolors.to_hex(color_names["orange"]),
    "CbG": mcolors.to_hex(color_names["purple"]),
    "GiG": mcolors.to_hex(color_names["pink"]),
    "ALL": mcolors.to_hex(color_names["light-green"]),
}

# ## Figure 2 Generative Model - AUROC

gen_model_performance_df = pd.read_csv(
    Path("../generative_model_training/output/generative_model_performance.tsv"),
    sep="\t",
)
gen_model_performance_df.lf_num = pd.Categorical(
    gen_model_performance_df.lf_num.tolist(),
    categories=["0", "1", "6", "11", "16", "All"],
)
gen_model_performance_df >> ply.slice_rows(10)

g = (
    p9.ggplot(
        gen_model_performance_df
        >> ply.query("data_source=='abstract'")
        >> ply.query("model=='test'")
    )
    + p9.aes(
        x="lf_num",
        y="auroc_mean",
        ymin="auroc_lower_ci",
        ymax="auroc_upper_ci",
        group="label_source",
        color="label_source",
    )
    + p9.geom_point()
    + p9.geom_line()
    + p9.geom_errorbar()
    + p9.facet_wrap("~ prediction_label")
    + p9.theme_bw()
    + p9.theme(figure_size=(8, 6))
    + p9.scale_color_manual(values=color_map)
    + p9.labs(
        title="Test Set AUROC of Predicted Relations", color="Relation (LF) Source"
    )
    + p9.xlab("Number of Label Functions")
    + p9.ylab("AUROC")
)
g.save("output/figure_two.svg")
g.save("output/figure_two.png", dpi=300)
print(g)

# ## Figure 3 - Generative Model AUPR

g = (
    p9.ggplot(
        gen_model_performance_df
        >> ply.query("data_source=='abstract'")
        >> ply.query("model=='test'")
    )
    + p9.aes(
        x="lf_num",
        y="aupr_mean",
        ymin="aupr_lower_ci",
        ymax="aupr_upper_ci",
        group="label_source",
        color="label_source",
    )
    + p9.geom_point()
    + p9.geom_line()
    + p9.geom_errorbar()
    + p9.facet_wrap("~ prediction_label")
    + p9.theme_bw()
    + p9.theme(figure_size=(8, 6))
    + p9.scale_color_manual(values=color_map)
    + p9.labs(
        title="Test Set AUPR of Predicted Relations", color="Relation (LF) Source"
    )
    + p9.xlab("Number of Label Functions")
    + p9.ylab("AUPR")
)
g.save("output/figure_three.svg")
g.save("output/figure_three.png", dpi=300)
print(g)

# ## Figure 4 - Generative Model All Label Functions

gen_model_performance_all_df = pd.read_csv(
    Path(
        "../generative_model_training_all_labels/output/generative_model_all_lf_performance.tsv"
    ),
    sep="\t",
)
gen_model_performance_all_df >> ply.slice_rows(10)

beginning_point_df = (
    gen_model_performance_df
    >> ply.query("lf_num=='0'")
    >> ply.query("data_source=='abstract'")
    >> ply.query("label_source==prediction_label")
    >> ply.define(label_source='"ALL"')
)
beginning_point_df

gen_model_performance_all_df = (
    gen_model_performance_all_df
    >> ply.call(".append", gen_model_performance_df)
    >> ply.query("data_source=='abstract'")
    >> ply.call(".append", beginning_point_df)
    >> ply.define(lf_num=ply.expressions.if_else("lf_num=='All'", 99, "lf_num"))
    >> ply.define(lf_num="lf_num.astype(int)")
    >> ply.define(
        lf_num=ply.expressions.case_when(
            {
                "lf_num==1": "lf_num",
                "lf_num==6": 33,
                "lf_num==11": 65,
                "lf_num==16": 97,
                True: "lf_num",
            }
        )
    )
)
gen_model_performance_all_df.lf_num = pd.Categorical(
    gen_model_performance_all_df.lf_num.tolist(), categories=[0, 1, 33, 65, 97, 99]
)
gen_model_performance_all_df >> ply.slice_rows(10)

g = (
    p9.ggplot(
        gen_model_performance_all_df
        >> ply.query("label_source==prediction_label|label_source=='ALL'")
        >> ply.query("model=='test'")
    )
    + p9.aes(
        x="lf_num",
        y="aupr_mean",
        ymin="aupr_lower_ci",
        ymax="aupr_upper_ci",
        group="label_source",
        color="label_source",
    )
    + p9.geom_point()
    + p9.geom_line()
    + p9.geom_errorbar()
    + p9.facet_wrap("~ prediction_label")
    + p9.theme_bw()
    + p9.theme(figure_size=(8, 6))
    + p9.scale_color_manual(values=color_map)
    + p9.labs(
        title="Test Set AUPR of Predicted Relations", color="Relation (LF) Source"
    )
    + p9.xlab("Number of Label Functions")
    + p9.ylab("AUPR")
)
g.save("output/figure_four.svg")
g.save("output/figure_four.png", dpi=300)
print(g)

g = (
    p9.ggplot(
        gen_model_performance_all_df
        >> ply.query("label_source==prediction_label|label_source=='ALL'")
        >> ply.query("model=='test'")
    )
    + p9.aes(
        x="lf_num",
        y="auroc_mean",
        ymin="auroc_lower_ci",
        ymax="auroc_upper_ci",
        group="label_source",
        color="label_source",
    )
    + p9.geom_point()
    + p9.geom_line()
    + p9.geom_errorbar()
    + p9.facet_wrap("~ prediction_label")
    + p9.theme_bw()
    + p9.theme(figure_size=(8, 6))
    + p9.scale_color_manual(values=color_map)
    + p9.labs(
        title="Test Set AUROC of Predicted Relations", color="Relation (LF) Source"
    )
    + p9.xlab("Number of Label Functions")
    + p9.ylab("AUROC")
)
g.save("output/figure_five.svg")
g.save("output/figure_five.png", dpi=300)
print(g)

# # Discriminator Model Figures

# ## Figure 5 - Discriminator Model vs Generative Model

disc_performance_df = pd.read_csv(
    Path("../discriminative_model_training/output/all_total_lf_performance.tsv"),
    sep="\t",
)
disc_performance_df.lf_num = pd.Categorical(
    disc_performance_df.lf_num.tolist(), categories=["0", "1", "6", "11", "16", "All"]
)
disc_performance_df

g = (
    p9.ggplot(disc_performance_df >> ply.query("dataset=='test'"))
    + p9.aes(
        x="lf_num",
        y="auroc_mean",
        ymin="auroc_lower_ci",
        ymax="auroc_upper_ci",
        group="model",
        color="label_source",
        linetype="model",
    )
    + p9.geom_point()
    + p9.geom_line()
    + p9.geom_errorbar()
    + p9.scale_color_manual(values=color_map)
    + p9.facet_wrap("~ prediction_label")
    + p9.theme_bw()
    + p9.labs(title="Test Set (AUROC)")
    + p9.theme(figure_size=(8, 6))
)
g.save("output/figure_six.svg")
g.save("output/figure_six.png", dpi=300)
print(g)

g = (
    p9.ggplot(disc_performance_df >> ply.query("dataset=='test'"))
    + p9.aes(
        x="lf_num",
        y="aupr_mean",
        ymin="aupr_lower_ci",
        ymax="aupr_upper_ci",
        group="model",
        color="label_source",
        linetype="model",
    )
    + p9.geom_point()
    + p9.geom_line()
    + p9.geom_errorbar()
    + p9.scale_color_manual(values=color_map)
    + p9.facet_wrap("~ prediction_label")
    + p9.theme_bw()
    + p9.labs(title="Test Set (AUPR)")
    + p9.theme(figure_size=(8, 6))
)
g.save("output/figure_seven.svg")
g.save("output/figure_seven.png", dpi=300)
print(g)

# # Edge Prediction

edge_recall_df = (
    pd.concat(
        [
            pd.read_csv(recall_file_path, sep="\t", index_col=0)
            for recall_file_path in Path("../edge_prediction_experiment/output/").rglob(
                "*edge_recall.tsv"
            )
        ]
    )
    >> ply.call(".reset_index")
    >> ply.select("-index")
)
edge_recall_df

g = (
    p9.ggplot(edge_recall_df, p9.aes(x="relation", y="edges", fill="in_hetionet"))
    + p9.geom_col(position="dodge")
    + p9.geom_text(
        p9.aes(
            label=(
                edge_recall_df.apply(
                    lambda x: f"{x['edges']:,} ({x['recall']:.0%})"
                    if not math.isnan(x["recall"])
                    else f"{x['edges']:,}",
                    axis=1,
                )
            )
        ),
        position=p9.position_dodge(width=1),
        size=7,
        va="bottom",
    )
    + p9.scale_y_log10()
    + p9.theme_seaborn("white")
    + p9.theme(
        axis_text_y=p9.element_blank(),
        axis_ticks_major=p9.element_blank(),
        axis_ticks_minor=p9.element_blank(),
        rect=p9.element_blank(),
        axis_line=p9.element_blank(),
        figure_size=(8, 6),
    )
    + p9.labs(title="Hetionet Edge Recall", fill="In Hetionet")
)
g.save("output/figure_eight.svg")
g.save("output/figure_eight.png", dpi=300)
print(g)
