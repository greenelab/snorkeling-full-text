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

# # Model Calibration Performance

# +
from pathlib import Path

import pandas as pd
import plotnine as p9
import plydata as ply
# -

# # Load the Calibration Performance Results

calibration_files = list(Path("output/calibration").rglob("*tsv"))

calibration_results_df = pd.concat(
    [
        pd.read_csv(file, sep="\t", index_col=0)
        >> ply.define(edge_type=f'{file.stem.split("_")[0]}')
        for file in calibration_files
    ]
)
calibration_results_df >> ply.slice_rows(10)

# # Plot the calibration Plots

g = (
    p9.ggplot(
        calibration_results_df,
        p9.aes(
            x="true_proportion",
            y="prediction_proportion",
            group="label",
            shape="label",
            color="edge_type",
        ),
    )
    + p9.geom_point()
    + p9.geom_line()
    + p9.scale_color_brewer(type="qual", palette=2)
    + p9.scale_y_continuous(limits=[0, 1])
    + p9.scale_x_continuous(limits=[0, 1])
    + p9.geom_segment(
        x=0, xend=1, y=0, yend=1, color="black", linetype="dashed", alpha=0.05
    )
    + p9.facet_wrap("~ edge_type")
)
print(g)
