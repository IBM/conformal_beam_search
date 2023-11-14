#
# Copyright IBM Corp. 2020-
# SPDX-License-Identifier: Apache2.0
#
"""Plotting code with hardcoded settings used in the paper.
NB: Functions in this module return g, a sns.JointGrid object,
which can be adjusted if current settings are unsuitable by accessing:
- g.fig (pyplot Figure)
- g.ax_joint (pyplot Axes for the main plot)
- g.ax_marg_x (pyplot Axes for the top-plot
To view the figure again, use g.fig.show() or in Jupyter:
```
from IPython import display
display(g.fig)
```
"""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# noinspection PyUnresolvedReferences
from confbeam_experiments.dyn_beams.analysis import collect_result_from_dir
from confbeam_experiments.dyn_beams.__main__ import ExperimentName

pretty_names = {"additions": "Additions", "rxn": "RXN"}
steps = {"additions": 5, "rxn": 60}


def gen_cov_per_len_plot(
    experiment: ExperimentName, res_df: pd.DataFrame
) -> sns.JointGrid:
    max_decode = 5
    plot_df = (
        res_df.groupby(["alpha", "rep", "L"])["in_beam"]
        .agg(["mean", "count"])
        .reset_index()
    )
    plot_df = plot_df.merge(
        plot_df.groupby(["alpha", "rep"])["count"]
        .sum()
        .rename("count_total")
        .reset_index()
    )
    plot_df = plot_df.assign(L_frac=plot_df["count"] / plot_df.count_total)
    plot_df = plot_df.assign(cl=1 - plot_df.alpha)

    hist_df = res_df

    n_mcs = set(res_df.groupby("alpha").rep.unique().apply(len).to_list())
    assert len(n_mcs) == 1, "All CLs must have the same number of repetitions"
    n_mc = n_mcs.pop()

    g = sns.JointGrid(ratio=7, height=4, space=0.1, marginal_ticks=True)
    sns.lineplot(
        plot_df,
        x="L",
        y="mean",
        hue="cl",
        palette="colorblind",
        errorbar=("sd", 1 / n_mc**0.5),
        ax=g.ax_joint,
    )

    sns.lineplot(
        plot_df,
        x="L",
        y=(1 - plot_df["alpha"]) ** max_decode,
        hue="cl",
        palette="colorblind",
        ls="--",
        ax=g.ax_joint,
        legend=False,
        zorder=-1000,
    )

    sns.histplot(
        hist_df,
        x="L",
        ax=g.ax_marg_x,
        discrete=True,
        element="step",
        cumulative=True,
        fill=False,
        stat="probability",
        common_norm=False,
        legend=False,
    )

    g.ax_joint.set(
        xlabel="Sequence Length",
        ylabel="Sequence-Level Coverage",
    )
    g.ax_marg_x.set(ylim=[None, None], ylabel="Frac.")

    sns.move_legend(
        g.ax_joint, "lower right", title="$1-\\alpha$", frameon=True, framealpha=1
    )

    g.ax_joint.xaxis.set_major_locator(plt.MultipleLocator(1))

    sns.despine(ax=g.ax_joint, top=False, right=False, left=False, bottom=False)
    sns.despine(ax=g.ax_marg_x, top=False, right=False, left=False, bottom=False)
    g.ax_marg_y.remove()

    fig = g.fig
    fig.suptitle(
        f"{pretty_names[experiment]}: Dynamic Beams ($\\leq {steps[experiment]}$ steps)",
        y=0.94,
    )
    fig.set_size_inches(4.5, 4.0)
    fig.tight_layout(h_pad=0.5)
    fig.set_dpi(150)
    return g


def gen_calibration_plot(
    experiment: ExperimentName, res_df: pd.DataFrame
) -> sns.JointGrid:
    plot_df = res_df
    plot_df = plot_df.assign(cl=1 - plot_df.alpha)

    n_mcs = set(res_df.groupby("alpha").rep.unique().apply(len).to_list())
    assert len(n_mcs) == 1, "All CLs must have the same number of repetitions"
    n_mc = n_mcs.pop()

    g = sns.JointGrid(ratio=7, height=4, space=0.1, marginal_ticks=True)
    sns.lineplot(
        plot_df,
        x="oracle_size",
        y="beam_size",
        hue="cl",
        palette="colorblind",
        errorbar=("sd", 1 / n_mc**0.5),
        ax=g.ax_joint,
    )

    sns.histplot(
        plot_df,
        x="oracle_size",
        hue="cl",
        discrete=True,
        ax=g.ax_marg_x,
        palette="colorblind",
        element="step",
        cumulative=True,
        fill=False,
        stat="probability",
        common_norm=False,
        legend=False,
    )

    g.ax_joint.set(
        xlabel="Oracle Size",
        ylabel="Final Beam Size",
    )
    g.ax_marg_x.set(ylim=[None, None], ylabel="Frac.")

    sns.move_legend(
        g.ax_joint, "upper left", title="$1-\\alpha$", frameon=True, framealpha=1
    )

    g.ax_joint.xaxis.set_major_locator(plt.MultipleLocator(1))

    sns.despine(ax=g.ax_joint, top=False, right=False, left=False, bottom=False)
    sns.despine(ax=g.ax_marg_x, top=False, right=False, left=False, bottom=False)
    g.ax_marg_y.remove()

    fig = g.fig
    fig.suptitle(
        f"{pretty_names[experiment]}: Dynamic Beams ($\\leq {steps[experiment]}$ steps)",
        y=0.94,
    )
    fig.set_size_inches(4.5, 4.0)
    fig.tight_layout(h_pad=0.2)
    fig.set_dpi(150)
    return g
