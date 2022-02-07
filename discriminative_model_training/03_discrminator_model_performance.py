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

# # Discriminator Model Trained on Weakly Supervised Data

# +
from pathlib import Path

import pandas as pd
import plotnine as p9
import plydata as ply
import scipy.stats
# -

# # Load the Data

# ## Generative Model

gen_model_performance_dfs = pd.concat(
    [
        pd.read_csv(str(df_path), sep="\t")
        >> ply.query("data_source=='abstract'")
        >> ply.define(lf_num=ply.expressions.if_else("lf_num > 16", 99, "lf_num"))
        >> ply.define(
            lf_num=ply.expressions.case_when(
                {
                    "lf_num==1": "lf_num",
                    "lf_num==6": "lf_num",
                    "lf_num==11": "lf_num",
                    "lf_num==16": "lf_num",
                    "lf_num==99": "lf_num",
                    True: 0,
                }
            )
        )
        >> ply.define(
            label_source=ply.expressions.if_else(
                'label_source.str.contains("baseline")',
                'label_source.str.replace("_baseline", "")',
                "label_source",
            )
        )
        >> ply.define(prediction_label='"' + df_path.stem.split("_")[0] + '"')
        >> ply.select("-sampled_lf_name", "-bce_loss", "-data_source")
        >> ply.rename(dataset="model")
        >> ply.define(model='"Gen"')
        for df_path in (
            Path("../generative_model_training/output/performance").rglob(
                "*performance.tsv"
            )
        )
    ]
)
gen_model_performance_dfs

# ## Discriminative Model

disc_model_performance_dfs = pd.concat(
    [
        pd.read_csv(str(df_path), sep="\t")
        >> ply.call(".dropna")
        >> ply.define(model='"Disc"')
        >> ply.define(lf_num=ply.expressions.if_else("lf_num > 16", 99, "lf_num"))
        >> ply.define(
            lf_num=ply.expressions.case_when(
                {
                    "lf_num==1": "lf_num",
                    "lf_num==6": "lf_num",
                    "lf_num==11": "lf_num",
                    "lf_num==16": "lf_num",
                    "lf_num==99": "lf_num",
                    True: 0,
                }
            )
        )
        >> ply.define(
            label_source=ply.expressions.if_else(
                'label_source.str.contains("baseline")',
                'label_source.str.replace("_baseline", "")',
                "label_source",
            )
        )
        >> ply.query("prediction_edge == label_source")
        >> ply.rename(aupr="AUPR", auroc="AUROC", prediction_label="prediction_edge")
        for df_path in Path().rglob("output/*total_lf_performance.tsv")
    ]
)
disc_model_performance_dfs

combined_models_df = gen_model_performance_dfs.append(disc_model_performance_dfs)
combined_models_df


# # Clean up the data and add Confidence Intervals

# +
def upper_ci(x):
    x.mean() + (scipy.stats.sem(x) * scipy.stats.t.ppf((1 + 0.95) / 2.0, len(x) - 1))


def lower_ci(x):
    x.mean() - (scipy.stats.sem(x) * scipy.stats.t.ppf((1 + 0.95) / 2.0, len(x) - 1))


# +
performance_ci_df = (
    combined_models_df
    >> ply.query("label_source == prediction_label")
    >> ply.group_by("lf_num", "label_source", "prediction_label", "dataset", "model")
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
    >> ply.ungroup()
    >> ply.define(lf_num=ply.expressions.if_else("lf_num==99", '"All"', "lf_num"))
    >> ply.distinct()
    >> ply.arrange("lf_num")
)

performance_ci_df.lf_num = pd.Categorical(
    performance_ci_df.lf_num.tolist(),
    categories=["0", "1", "6", "11", "16", "All"],
    ordered=True,
)

performance_ci_df
# -

# # Plot Performance

# ## Train

g = (
    p9.ggplot(performance_ci_df >> ply.query("dataset=='tune'"))
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
    + p9.scale_color_brewer(type="qual", palette=2)
    + p9.facet_wrap("~ prediction_label")
    + p9.scale_y_continuous(limits=[0, 1])
    + p9.theme_seaborn("white")
    + p9.labs(title="Tune Set (AUROC)")
    + p9.theme(figure_size=(10, 10))
)
print(g)

g = (
    p9.ggplot(performance_ci_df >> ply.query("dataset=='tune'"))
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
    + p9.scale_color_brewer(type="qual", palette=2)
    + p9.facet_wrap("~ prediction_label")
    + p9.scale_y_continuous(limits=[0, 0.8])
    + p9.theme_seaborn("white")
    + p9.labs(title="Tune Set (AUPR)")
    + p9.theme(figure_size=(10, 10))
)
print(g)

# ## Test

g = (
    p9.ggplot(performance_ci_df >> ply.query("dataset=='test'"))
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
    + p9.scale_color_brewer(type="qual", palette=2)
    + p9.facet_wrap("~ prediction_label")
    + p9.theme_seaborn("white")
    + p9.labs(title="Test Set (AUROC)")
    + p9.theme(figure_size=(10, 10))
)
print(g)

g = (
    p9.ggplot(performance_ci_df >> ply.query("dataset=='test'"))
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
    + p9.scale_color_brewer(type="qual", palette=2)
    + p9.facet_wrap("~ prediction_label")
    + p9.theme_seaborn("white")
    + p9.labs(title="Test Set (AUPR)")
    + p9.theme(figure_size=(10, 10))
)
print(g)
