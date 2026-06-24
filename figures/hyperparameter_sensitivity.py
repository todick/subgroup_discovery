import click
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns
import sys
from pathlib import Path

sys.path.append("../")

from plot_config import (
    COLOR_PALETTE,
    LABELFONTSIZE,
    LEGENDFONTSIZE,
    TICKFONTSIZE,
    XLABELPAD,
    YLABELPAD,
    MARKERLINEWIDTH,
    init_plot_style,
    load_experiment_results,
    darken_color,
)

init_plot_style(style="whitegrid", font_weight="bold", axes_linewidth=0.5)

DEFAULT_METHODS = ["beam-mean", "beam-kl", "dfs-mean", "rsd", "syflow"]
METHOD_NAME_MAP = {
    "beam-mean": "\\textsc{BS-$\\mu$}",
    "beam-kl": "\\textsc{BS-kl}",
    "dfs-mean": "\\textsc{DFS-$\\mu$}",
    "rsd": "\\textsc{RSD}",
    "syflow": "\\textsc{Syflow}",
}
METHOD_MARKERS = {
    "\\textsc{BS-$\\mu$}": "s",
    "\\textsc{BS-kl}": "s",
    "\\textsc{DFS-$\\mu$}": "D",
    "\\textsc{RSD}": "o",
    "\\textsc{Syflow}": "X",
}
PARAM_LABELS = {
    "alpha": r"$\alpha$",
    "beam_width": "Beam Width",
    "rsd_min_support": r"Min.\ Support",
    "lambd": r"$\lambda$",
}
PARAM_ORDER = ["alpha", "lambd", "beam_width", "rsd_min_support"]
PARAM_LOGSCALE = {"beam_width", "rsd_min_support"}
PARAM_CATEGORICAL = {"lambd"}  # evenly-spaced ticks because lambd includes 0
RQ1_DEFAULTS = {
    "alpha": 0.5,
    "beam_width": 40,
    "rsd_min_support": 0.025,
    "lambd": 0.5,
}

output_dir = Path("figures/out")


def load_hyperparameter_data():
    base = Path("results/hyperparameter_sensitivity/hyperparameter_sensitivity")
    frames = []
    for method in DEFAULT_METHODS:
        folder = base / method / "workload"
        if not folder.exists():
            continue
        df = load_experiment_results(str(folder))
        if df.empty:
            continue
        df["method"] = METHOD_NAME_MAP[method]
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def fmt_tick(v):
    """Format a param value as a clean string (e.g. 20.0 â†’ '20', 0.025 â†’ '0.025')."""
    if v == int(v):
        return str(int(v))
    return str(v)


@click.group()
def cli():
    pass


@cli.command()
def plot_sensitivity():
    """Plot hyperparameter sensitivity as lineplots, one subplot per hyperparameter."""
    df = load_hyperparameter_data()
    if df.empty:
        print("No data found.")
        return

    df["mean_f1"] = df[["f1_0", "f1_1", "f1_2"]].sum(axis=1) / df["n_groups"]

    agg_df = (
        df.groupby(["method", "param_name", "param_value", "n_groups"], observed=True, as_index=False)
        .agg(mean_f1=("mean_f1", "mean"), std_f1=("mean_f1", "std"))
    )
    agg_df["std_f1"] = agg_df["std_f1"].fillna(0)

    params = [p for p in PARAM_ORDER if p in agg_df["param_name"].unique()]
    method_names = [METHOD_NAME_MAP[m] for m in DEFAULT_METHODS]
    color_palette = sns.color_palette(COLOR_PALETTE, len(method_names))
    method_colors = dict(zip(method_names, color_palette))

    output_dir.mkdir(parents=True, exist_ok=True)

    for n_groups in sorted(agg_df["n_groups"].unique()):
        ng_df = agg_df[agg_df["n_groups"] == n_groups]

        n_params = len(params)
        fig_width = max(6.0, 2.4 * n_params)
        fig_height = 2.2
        fig, axes = plt.subplots(1, n_params, figsize=(fig_width, fig_height), squeeze=False, sharey=True)
        axes = axes.flatten()

        legend_handles = {}

        for i, param in enumerate(params):
            ax = axes[i]
            param_df = ng_df[ng_df["param_name"] == param]
            methods_present = [m for m in method_names if m in param_df["method"].unique()]
            x_vals = sorted(param_df["param_value"].unique())
            categorical = param in PARAM_CATEGORICAL
            x_positions = list(range(len(x_vals))) if categorical else x_vals

            for method in methods_present:
                method_df = param_df[param_df["method"] == method].sort_values("param_value")
                color = method_colors[method]
                marker = METHOD_MARKERS[method]
                xs = [x_positions[x_vals.index(v)] for v in method_df["param_value"]] if categorical else method_df["param_value"]
                ax.errorbar(
                    xs,
                    method_df["mean_f1"],
                    yerr=method_df["std_f1"],
                    color=color,
                    marker=marker,
                    markersize=5,
                    linewidth=1.2,
                    markeredgecolor=darken_color(color),
                    markeredgewidth=MARKERLINEWIDTH,
                    markerfacecolor=(*color, 0.7),
                    elinewidth=0.7,
                    capsize=2.5,
                    capthick=0.7,
                    ecolor=darken_color(color),
                    label=method,
                )
                if method not in legend_handles:
                    legend_handles[method] = Line2D(
                        [0], [0],
                        marker=marker,
                        color=color,
                        linewidth=1.2,
                        markersize=5,
                        markerfacecolor=(*color, 0.7),
                        markeredgecolor=darken_color(color),
                        markeredgewidth=MARKERLINEWIDTH,
                    )

            if param in PARAM_LOGSCALE:
                ax.set_xscale("log")
                ax.set_xticks(x_vals)
                ax.set_xticklabels([fmt_tick(v) for v in x_vals], fontsize=TICKFONTSIZE - 1)
                ax.xaxis.set_minor_locator(plt.NullLocator())
            elif categorical:
                ax.set_xticks(x_positions)
                ax.set_xticklabels([fmt_tick(v) for v in x_vals], fontsize=TICKFONTSIZE - 1)
            else:
                ax.set_xticks(x_vals)
                ax.set_xticklabels([fmt_tick(v) for v in x_vals], fontsize=TICKFONTSIZE - 1)

            if param in RQ1_DEFAULTS:
                default_val = RQ1_DEFAULTS[param]
                xline = x_positions[x_vals.index(default_val)] if categorical else default_val
                ax.axvline(xline, color="black", linestyle="--", linewidth=0.9, alpha=0.6, zorder=0)

            ax.set_xlabel(PARAM_LABELS.get(param, param), fontsize=LABELFONTSIZE, labelpad=XLABELPAD)
            ax.set_ylabel("Mean F1" if i == 0 else "", fontsize=LABELFONTSIZE, labelpad=YLABELPAD)
            ax.set_ylim(0, 1.05)
            ax.set_yticks([0, 0.5, 1.0])
            ax.tick_params(axis="x", labelsize=TICKFONTSIZE - 1, bottom=True, width=0.5, length=3)
            ax.tick_params(axis="y", labelsize=TICKFONTSIZE, left=True, width=0.5, length=3)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color("black")
            ax.spines["bottom"].set_color("black")
            ax.grid(visible=True, which="major", linestyle="--", linewidth=0.5, alpha=0.7)

        rq1_handle = Line2D([0], [0], color="black", linestyle="--", linewidth=0.9, alpha=0.6)
        leg_handles = [legend_handles[m] for m in method_names if m in legend_handles] + [rq1_handle]
        leg_labels = [m for m in method_names if m in legend_handles] + ["Values for RQ1 \& RQ2"]

        fig.legend(
            leg_handles,
            leg_labels,
            fontsize=LEGENDFONTSIZE,
            loc="upper center",
            ncol=len(leg_handles),
            bbox_to_anchor=(0.5, 1.04),
            edgecolor="white",
            handletextpad=0.3,
            borderaxespad=0.3,
            columnspacing=1.2,
            handlelength=1.2,
        )

        plt.tight_layout()
        plt.subplots_adjust(top=0.84, wspace=0.18)

        out_path = output_dir / f"hyperparameter_sensitivity_n{n_groups}.pdf"
        plt.savefig(out_path, dpi=300, bbox_inches="tight", format="pdf")
        plt.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight", format="png")
        plt.close(fig)
        print(f"Saved to {out_path}")


if __name__ == "__main__":
    cli()


