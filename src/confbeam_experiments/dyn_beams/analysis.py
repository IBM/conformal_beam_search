#
# Copyright IBM Corp. 2020-
# SPDX-License-Identifier: Apache2.0
#
"""Post-processing and aggregation of dynamic conformal beam decoding experiments"""
import pickle
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd


def collect_result_from_dir(exp_dir: Path) -> pd.DataFrame:
    """Read saved lists of dataframes across repetitions and risks and return aggregated dataframe"""
    assert exp_dir.is_dir()
    result_files = exp_dir.glob("dynbeam_exp_result-*.pkl")
    all_results = []
    for rf in result_files:
        with open(rf, "rb") as pf:
            all_results.extend(pickle.load(pf))
    global_results = pd.concat(
        [prepare_for_global(r) for r in all_results], ignore_index=True
    )
    return global_results


def prepare_for_global(result_df: pd.DataFrame) -> pd.DataFrame:
    """Drop step-wise information to keep final beams at the last decoding step"""
    kept_columns = ["in_beam", "oracle_size", "beam_size", "L"]
    result_df = result_df.iloc[result_df.groupby("sample_idx")["step"].idxmax()]
    result_df = result_df[kept_columns]
    result_df = result_df.assign(alpha=result_df.attrs["alpha"])
    result_df = result_df.assign(rep=result_df.attrs["rep"])
    return result_df


def aggregate_global_results(global_results: pd.DataFrame) -> pd.DataFrame:
    """Obtain MC mean and uncertainty for each risk over repetitions"""
    global_results = global_results.assign(
        beam_factor=global_results["beam_size"] / global_results["oracle_size"]
    )
    metrics = ["in_beam", "beam_size", "beam_factor"]
    aggs = ["mean", "std", "count"]
    mean_metrics_df = global_results.groupby(["alpha", "rep"])[metrics].mean()

    agg_results = mean_metrics_df.groupby("alpha")[metrics].agg(aggs)
    for m in metrics:
        agg_results[(m, "unc")] = (
            agg_results[(m, "std")] / agg_results[(m, "count")] ** 0.5
        )

    columns_pretty_order = [(m, a) for m, a in product(metrics, aggs + ["unc"])]

    return agg_results[columns_pretty_order]


def format_metric(metric):
    """Internal: format (mean,unc) to latex M.EAN(U) with the right sigfigs"""

    def format_metric_(row):
        res = row[(metric, "mean")]
        unc = row[(metric, "unc")]
        digits_unc = np.ceil(-np.log(unc) / np.log(10)).astype(int)
        res_str = f"{str(res)[:2 + digits_unc]}({int(unc * 10 ** digits_unc)})"
        return res_str

    return format_metric_


def format_aggregated_results_latex(agg_results: pd.DataFrame, latex_out_buffer=None):
    """Create latex table as table 2. (per problem)"""
    latex_col_map = {
        "in_beam": "Coverage",
        "beam_size": r"$\left\langle\left|\beta(X)\right|\right\rangle$",
        "beam_factor": r"$\left\langle\left|\beta (X)\right| / \left|\beta_\text{O} (X)\right|\right\rangle$",
        "cl": r"$(1-\alpha)$",
    }

    agg_results_pretty = pd.DataFrame(
        [
            agg_results.apply(format_metric("in_beam"), axis=1).rename("in_beam"),
            agg_results.apply(format_metric("beam_size"), axis=1).rename("beam_size"),
            agg_results.apply(format_metric("beam_factor"), axis=1).rename(
                "beam_factor"
            ),
        ]
    ).T.reset_index()
    agg_results_pretty = agg_results_pretty.assign(cl=1 - agg_results_pretty.alpha)
    agg_results_pretty = agg_results_pretty.drop(columns="alpha")
    ordered_columns = ["cl", "in_beam", "beam_size", "beam_factor"]
    agg_results_pretty = agg_results_pretty[ordered_columns]

    agg_results_pretty = agg_results_pretty.rename_axis(latex_col_map["cl"]).rename(
        columns=latex_col_map
    )
    latex_result = agg_results_pretty.to_latex(
        buf=latex_out_buffer, float_format="%.3f", index=False
    )
    return latex_result
