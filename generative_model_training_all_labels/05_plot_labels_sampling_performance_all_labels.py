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

# # Plot Label Resampling Performance

# +
from pathlib import Path

import scipy.stats
import pandas as pd
import plotnine as p9
import plydata as ply
# -

# ## Gather performance dataset files and manipulate the dataframes

# These label function cutoffs are choosen to have evenly spaced
# sample sizes where 32 appears to be a sizeable amount to notice a difference
# and have quick run time
performance_dfs = pd.concat(
    [
        pd.read_csv(str(df_path), sep="\t")
        >> ply.define(lf_num=ply.expressions.if_else("lf_num > 97", 99, "lf_num"))
        >> ply.define(
            lf_num=ply.expressions.case_when(
                {
                    "lf_num==1": "lf_num",
                    "lf_num==33": "lf_num",
                    "lf_num==65": "lf_num",
                    "lf_num==97": "lf_num",
                    "lf_num==99": "lf_num",
                    True: 0,
                }
            )
        )
        >> ply.define(
            prediction_label='"' + df_path.stem.split("_")[1] + '"',
        )
        for df_path in Path("output").rglob("*tsv")
    ]
)
print(performance_dfs.shape)
performance_dfs.head()

# +
entity_labels = (
    performance_dfs
    >> ply.select("label_source")
    >> ply.distinct()
    >> ply.pull("label_source")
)

data_rows = []
for idx, row in performance_dfs >> ply.query("lf_num==0") >> ply.call(".iterrows"):

    for entity in entity_labels:

        if entity == row["label_source"]:
            continue

        data_rows.append(
            {
                "lf_num": 0,
                "aupr": row["aupr"],
                "auroc": row["auroc"],
                "bce_loss": row["bce_loss"],
                "sampeld_lf_name": row["sampled_lf_name"],
                "label_source": entity,
                "data_source": row["data_source"],
                "prediction_label": row["prediction_label"],
            }
        )


# +
def upper_ci(x):
    return x.mean() + (
        scipy.stats.sem(x) * scipy.stats.t.ppf((1 + 0.95) / 2.0, len(x) - 1)
    )


def lower_ci(x):
    return x.mean() - (
        scipy.stats.sem(x) * scipy.stats.t.ppf((1 + 0.95) / 2.0, len(x) - 1)
    )


# -

performance_ci_df = (
    performance_dfs
    >> ply.call(".append", pd.DataFrame.from_records(data_rows))
    >> ply.group_by(
        "lf_num", "label_source", "prediction_label", "data_source", "model"
    )
    >> ply.define(
        aupr_mean="mean(aupr)",
        aupr_upper_ci="upper_ci(aupr)",
        aupr_lower_ci="lower_ci(aupr)",
    )
    >> ply.define(
        auroc_mean="mean(auroc)",
        auroc_upper_ci="upper_ci(auroc)",
        auroc_lower_ci="lower_ci(auroc)",
    )
    >> ply.define(
        bce_mean="mean(bce_loss)",
        bce_upper_ci="upper_ci(bce_loss)",
        bce_lower_ci="lower_ci(bce_loss)",
    )
    >> ply.ungroup()
    >> ply.define(lf_sample=ply.expressions.if_else("lf_num==99", '"All"', "lf_num"))
    >> ply.select(
        "lf_num",
        "label_source",
        "prediction_label",
        "data_source",
        "model",
        "aupr_mean",
        "aupr_upper_ci",
        "aupr_lower_ci",
        "auroc_mean",
        "auroc_upper_ci",
        "auroc_lower_ci",
        "bce_mean",
        "bce_upper_ci",
        "bce_lower_ci",
    )
    >> ply.call(".dropna")
    >> ply.distinct()
)
performance_ci_df.lf_num = pd.Categorical(performance_ci_df.lf_num.astype(int).tolist())
performance_ci_df >> ply.call(
    "to_csv", "output/generative_model_all_lf_performance.tsv", sep="\t", index=False
)
performance_ci_df

# # Plot Different Dataset Performance

# ## Tune

g = (
    p9.ggplot(performance_ci_df >> ply.query("model=='tune'"))
    + p9.aes(
        x="lf_num",
        y="aupr_mean",
        ymin="aupr_lower_ci",
        ymax="aupr_upper_ci",
        group="label_source",
    )
    + p9.geom_point()
    + p9.geom_line()
    + p9.geom_errorbar()
    + p9.facet_wrap("~ prediction_label")
    + p9.theme_bw()
    + p9.theme(figure_size=(10, 10))
    + p9.scale_color_brewer(type="qual", palette=2)
)
print(g)

g = (
    p9.ggplot(performance_ci_df >> ply.query("model=='tune'"))
    + p9.aes(
        x="lf_num",
        y="auroc_mean",
        ymin="auroc_lower_ci",
        ymax="auroc_upper_ci",
        group="label_source",
    )
    + p9.geom_point()
    + p9.geom_line()
    + p9.geom_errorbar()
    + p9.facet_wrap("~ prediction_label")
    + p9.theme_bw()
    + p9.theme(figure_size=(10, 10))
    + p9.scale_color_brewer(type="qual", palette=2)
)
print(g)

g = (
    p9.ggplot(performance_ci_df >> ply.query("model=='tune'"))
    + p9.aes(
        x="lf_num",
        y="bce_mean",
        ymin="bce_lower_ci",
        ymax="bce_upper_ci",
        group="label_source",
    )
    + p9.geom_point()
    + p9.geom_line()
    + p9.geom_errorbar()
    + p9.facet_wrap("~ prediction_label")
    + p9.theme_bw()
    + p9.theme(figure_size=(10, 10))
    + p9.scale_color_brewer(type="qual", palette=2)
)
print(g)

# ## Test

g = (
    p9.ggplot(performance_ci_df >> ply.query("model=='test'"))
    + p9.aes(
        x="lf_num",
        y="aupr_mean",
        ymin="aupr_lower_ci",
        ymax="aupr_upper_ci",
        group="label_source",
    )
    + p9.geom_point()
    + p9.geom_line()
    + p9.geom_errorbar()
    + p9.facet_wrap("~ prediction_label")
    + p9.theme_bw()
    + p9.theme(figure_size=(10, 10))
    + p9.scale_color_brewer(type="qual", palette=2)
)
print(g)

g = (
    p9.ggplot(performance_ci_df >> ply.query("model=='test'"))
    + p9.aes(
        x="lf_num",
        y="auroc_mean",
        ymin="auroc_lower_ci",
        ymax="auroc_upper_ci",
        group="label_source",
    )
    + p9.geom_point()
    + p9.geom_line()
    + p9.geom_errorbar()
    + p9.facet_wrap("~ prediction_label")
    + p9.theme_bw()
    + p9.theme(figure_size=(10, 10))
    + p9.scale_color_brewer(type="qual", palette=2)
)
print(g)

g = (
    p9.ggplot(performance_ci_df >> ply.query("model=='test'"))
    + p9.aes(
        x="lf_num",
        y="bce_mean",
        ymin="bce_lower_ci",
        ymax="bce_upper_ci",
        group="label_source",
    )
    + p9.geom_point()
    + p9.geom_line()
    + p9.geom_errorbar()
    + p9.facet_wrap("~ prediction_label")
    + p9.theme_bw()
    + p9.theme(figure_size=(10, 10))
    + p9.scale_color_brewer(type="qual", palette=2)
)
print(g)
