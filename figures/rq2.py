import click
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import pymannkendall as mk
import seaborn as sns
import sys
from pathlib import Path
from typing import List
sys.path.append("../")

from plot_config import (
    COLOR_PALETTE,
    LABELFONTSIZE,
    LEGENDFONTSIZE,
    TICKFONTSIZE,
    LEGENDTITLEFONTSIZE,
    XLABELPAD,
    YLABELPAD,
    MARKERLINEWIDTH,
    init_plot_style,
    load_experiment_results,
    darken_color,
    set_scatter_edgecolors,
    get_edge_colors
)

# Initialize plotting style
init_plot_style(style="whitegrid", font_weight="bold", axes_linewidth=0.5)

DEFAULT_METHODS = ["beam-mean", "beam-kl", "dfs-mean", "rsd", "syflow"]
DEFAULT_METHOD_NAMES = ["\\textsc{BS-$\\mu$}", "\\textsc{BS-kl}", "\\textsc{DFS-$\\mu$}", "\\textsc{RSD}", "\\textsc{Syflow}"]
CART_METHODS = DEFAULT_METHODS + ["cart"]
CART_METHOD_NAMES = DEFAULT_METHOD_NAMES + ["\\textsc{CART}"]
METHOD_MARKERS = {
    "\\textsc{BS-$\\mu$}": "s",
    "\\textsc{BS-kl}": "s",
    "\\textsc{DFS-$\\mu$}": "D",
    "\\textsc{RSD}": "o",
    "\\textsc{Syflow}": "X",
    "\\textsc{CART}": "^",
}
output_dir = Path("figures/out")


def configure_plot_settings(with_cart=False):
    global methods, method_names, output_dir
    methods = CART_METHODS.copy() if with_cart else DEFAULT_METHODS.copy()
    method_names = CART_METHOD_NAMES.copy() if with_cart else DEFAULT_METHOD_NAMES.copy()
    output_dir = Path("figures/out/full") if with_cart else Path("figures/out")


def get_method_markers(active_method_names):
    return [METHOD_MARKERS[method_name] for method_name in active_method_names]


def get_output_path(filename):
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / filename

def load_experiment_results_multi_method(experiment_path, methods = ["syflow", "rsd", "beam-kl", "beam-mean", "dfs-mean"]):
    frames = []
    for method in methods:
        folder = f"results/RQ2/{method}/{experiment_path}"
        df_method = load_experiment_results(folder)
        if df_method.empty:
            continue
        df_method["method"] = method
        df_method["method"] = df_method["method"].replace({
            "syflow": "\\textsc{Syflow}",
            "rsd": "\\textsc{RSD}",
            "beam-kl": "\\textsc{BS-kl}",
            "beam-mean": "\\textsc{BS-$\\mu$}",
            "dfs-mean": "\\textsc{DFS-$\\mu$}",
            "cart": "\\textsc{CART}",
        })
        frames.append(df_method)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

@click.group()
def cli():
    configure_plot_settings(with_cart=False)

def plot_scalability(experiment_path, target_column):
    global methods, method_names
    df = load_experiment_results_multi_method(experiment_path, methods)
    df["method"] = pd.Categorical(df["method"], categories=method_names, ordered=True)
    color_palette = sns.color_palette(COLOR_PALETTE, len(methods))
    method_markers = get_method_markers(method_names)

    df_runtime = df[["seed", "method", target_column, "runtime"]].copy()
    # Add a dummy row for each method and target combination
    for method in method_names:
        for target in df_runtime[target_column].unique():
            for seed in df_runtime["seed"].unique():
                if not ((df_runtime["seed"] == seed) & (df_runtime["method"] == method) & (df_runtime[target_column] == target)).any():
                    df_runtime = pd.concat([df_runtime, pd.DataFrame({
                        "seed": [seed],
                        "method": [method],
                        target_column: [target],
                        "runtime": [14400],
                    })], ignore_index=True)

    runtime_plot_df = (
        df_runtime.groupby(["method", target_column], observed=True, as_index=False)
        .agg(runtime=("runtime", "mean"))
    )

    fig = plt.figure(figsize=(6, 3.25))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
    
    def plot_method_series(ax, subset, method_index, x_column, y_column):
        ax.plot(
            subset[x_column],
            subset[y_column],
            color=color_palette[method_index],
            linestyle="--",
            linewidth=1.2,
            alpha=0.5,
            marker=method_markers[method_index],
            markersize=6,
            markerfacecolor=(*color_palette[method_index], 0.7),
            markeredgecolor=darken_color(color_palette[method_index]),
            markeredgewidth=MARKERLINEWIDTH,
        )

    # Top-left plot: Runtime vs Number of Configuration Options (0-300 seconds)
    ax1 = fig.add_subplot(gs[0, 0])
    for i, method in enumerate(method_names):
        subset = runtime_plot_df[runtime_plot_df["method"] == method].sort_values(target_column)
        if subset["runtime"].max() > 600:
            plot_method_series(ax1, subset, i, target_column, "runtime")
    ax1.set_xscale("log")
    ax1.set_xlim(df[target_column].min(), df[target_column].max()*1.1)
    ax1.set_ylim(0, 7300)
    ax1.set_yticks([0, 3600, 7200])
    ax1.set_xlabel("")
    ax1.set_ylabel("")
    ax1.tick_params(axis="x", labelsize=TICKFONTSIZE, bottom=True, width=0.5, length=3)
    ax1.tick_params(axis="y", labelsize=TICKFONTSIZE, left=True, width=0.5, length=3)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_color("black")
    ax1.spines["bottom"].set_color("black")
    ax1.grid(visible=True, which="major", linestyle="--", linewidth=0.5, alpha=0.7)
    ax1.set_xticklabels([])
    
    # Bottom-left plot: Runtime vs Number of Configuration Options (0-3600 seconds)
    ax2 = fig.add_subplot(gs[1, 0])
    for i, method in enumerate(method_names):
        subset = runtime_plot_df[runtime_plot_df["method"] == method].sort_values(target_column)
        if subset["runtime"].max() < 600:
            plot_method_series(ax2, subset, i, target_column, "runtime")
    ax2.set_xscale("log")
    ax2.set_xlim(df[target_column].min(), df[target_column].max()*1.1)
    ax2.set_ylim(0, 370)
    ax2.set_yticks([0, 180, 360])
    ax2.tick_params(axis="x", labelsize=TICKFONTSIZE, bottom=True, width=0.5, length=3)
    ax2.tick_params(axis="y", labelsize=TICKFONTSIZE, left=True, width=0.5, length=3)
    if target_column == "n_features":
        ax2.set_xlabel("$|\\CMcal{O}|$", fontsize=LABELFONTSIZE, labelpad=XLABELPAD)
    else:
        ax2.set_xlabel("$|\\CMcal{C}|$", fontsize=LABELFONTSIZE, labelpad=XLABELPAD)
    ax2.set_ylabel("")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_color("black")
    ax2.spines["bottom"].set_color("black")
    ax2.grid(visible=True, which="major", linestyle="--", linewidth=0.5, alpha=0.7)

    # Right plot: F1 vs Number of Configuration Options
    df_scatter = df[["seed", "method", target_column, "f1"]].copy()
    df_scatter["identified"] = df.groupby(["method", target_column])["f1"].transform(lambda x: (x == 1.0).sum())    
    df_scatter = df_scatter.groupby(["method", target_column]).filter(lambda x: len(x) == 100)
    df_scatter = df_scatter.groupby(["method", target_column]).agg({"identified": "max"}).reset_index()
    ax3 = fig.add_subplot(gs[:, 1])  # Span both rows
    # add dashed lines and markers for each method
    for j, method in enumerate(method_names):
        subset = df_scatter[df_scatter["method"] == method].sort_values(target_column)
        if len(subset) > 0:
            plot_method_series(ax3, subset, j, target_column, "identified")

    if target_column == "n_features":
        ax3.set_xlabel("$|\\CMcal{O}|$", fontsize=LABELFONTSIZE, labelpad=XLABELPAD)
    else:
        ax3.set_xlabel("$|\\CMcal{C}|$", fontsize=LABELFONTSIZE, labelpad=XLABELPAD)
    ax3.set_ylabel(r"\% Identified", fontsize=LABELFONTSIZE, labelpad=YLABELPAD)
    ax3.set_xscale("log")
    ax3.set_xlim(df[target_column].min(), df[target_column].max()*1.1)
    ax3.set_ylim(0, 105)
    ax3.set_yticks([0, 25, 50, 75, 100])
    ax3.tick_params(axis="x", labelsize=TICKFONTSIZE, bottom=True, width=0.5, length=3)
    ax3.tick_params(axis="y", labelsize=TICKFONTSIZE, left=True, width=0.5, length=3)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.spines["left"].set_color("black")
    ax3.spines["bottom"].set_color("black")
    ax3.grid(visible=True, which="major", linestyle="--", linewidth=0.5, alpha=0.7)
    legend_handles = []
    for index, method in enumerate(method_names):
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker=method_markers[index],
                color=color_palette[index],
                linestyle="--",
                linewidth=1.2,
                markersize=6,
                markerfacecolor=(*color_palette[index], 0.7),
                markeredgecolor=darken_color(color_palette[index]),
                markeredgewidth=MARKERLINEWIDTH,
            )
        )
    fig.legend(
        legend_handles,
        method_names,
        title="",
        fontsize=LEGENDFONTSIZE-1,
        title_fontsize=LEGENDTITLEFONTSIZE,
        loc="upper center",
        ncol=5,
        bbox_to_anchor=(0.5, 1.03),
        edgecolor="white",
        handlelength=0.7,
        handletextpad=0.2,
        columnspacing=0.7,
    )
    # add some text on the left side of the plot without scaling any of the axes
    fig.text(
        0.04, 0.57, "Runtime (s)", ha="center", va="center", rotation=90,
        fontsize=LABELFONTSIZE, color="black"
    )
    
    if ax1.get_legend() is not None:
        ax1.get_legend().remove()
    if ax2.get_legend() is not None:
        ax2.get_legend().remove()
    if ax3.get_legend() is not None:
        ax3.get_legend().remove()

    plt.tight_layout()
    plt.subplots_adjust(top=0.87, wspace=0.35, hspace=0.4)
    plt.savefig(get_output_path(f"rq2_scalability_{target_column}.pdf"), dpi=300, bbox_inches="tight", format="pdf")
    plt.close(fig)

@cli.command()
@click.option("--with-cart", is_flag=True, default=False, help="Include CART and write output to figures/out/cart.")
def plot_scalability_features(with_cart):
    '''Plot scalability for number of features.'''
    configure_plot_settings(with_cart)
    plot_scalability("scalability_features/workload", "n_features")

@cli.command()
@click.option("--with-cart", is_flag=True, default=False, help="Include CART and write output to figures/out/cart.")
def plot_scalability_samples(with_cart):
    '''Plot scalability for number of samples.'''
    configure_plot_settings(with_cart)
    plot_scalability("scalability_samples/workload", "n_samples")


@cli.command()
@click.option("--with-cart", is_flag=True, default=False, help="Include CART and write output to figures/out/cart.")
def plot_sampling(with_cart):
    """Plot sampling for all case studies."""
    global methods, method_names
    configure_plot_settings(with_cart)
    df_all = load_experiment_results_multi_method("sampling/distance_based", methods)
    df_all["method"] = pd.Categorical(df_all["method"], categories=method_names, ordered=True)
    df_all["sampling_strategy"] = df_all["sampling_strategy"].astype(str)
    df_all["sample_size"] = df_all["sample_size"].astype(int)

    # Show all available systems
    available_systems = sorted(df_all["casestudy"].unique())
    
    df = df_all[df_all["casestudy"].isin(available_systems)].copy()
    df["casestudy"] = pd.Categorical(df["casestudy"], categories=available_systems, ordered=True)

    # Use df_all to determine strategies (so they're available even when showing only average)
    all_strategies = set(df_all["sampling_strategy"].unique())
    twise_order = sorted(
        [strategy for strategy in all_strategies if strategy.endswith("-wise")],
        key=lambda strategy: int(strategy.split("-")[0]) if strategy.split("-")[0].isdigit() else 0,
    )
    random_order = sorted(
        [strategy for strategy in all_strategies if strategy.startswith("random_")],
        key=lambda strategy: int(strategy.split("_", 1)[1]) if strategy.split("_", 1)[1].isdigit() else 0,
    )
    has_full = "full" in all_strategies

    strategy_labels = {
        strategy: (
            "$\\CMcal{C}$"
            if strategy == "full"
            else strategy.split("_", 1)[1]
            if strategy.startswith("random_")
            else strategy.split("-", 1)[0]
        )
        for strategy in all_strategies
    }

    subsets: List[tuple[str, List[str], str]] = [
        ("t-wise", twise_order, "Coverage"),
        ("random", random_order, "Sample Size"),
    ]

    # Add 1 row for average
    n_rows = len(available_systems) + 1
    row_height = 1.125
    fig, axes = plt.subplots(
        n_rows,
        3,
        figsize=(6, row_height * n_rows + 0.75),  # Dynamic height based on number of systems
        sharex="col",
        sharey=True,
        squeeze=False,
        gridspec_kw={"width_ratios": [1.0, 1.0, 0.3]},
    )

    color_palette = sns.color_palette(COLOR_PALETTE, len(method_names))
    method_markers = get_method_markers(method_names)
    legend_handles = None
    legend_labels = None

    for row_index, casestudy in enumerate(available_systems):
        for col_index, (subset_name, subset_order, axis_label) in enumerate(subsets):
            ax = axes[row_index][col_index]
            casestudy_df = df[df["casestudy"] == casestudy]
            subset_df = casestudy_df[casestudy_df["sampling_strategy"].isin(subset_order)].copy()
            if subset_df.empty or not subset_order:
                ax.set_visible(False)
                continue

            subset_df["sampling_strategy"] = pd.Categorical(
                subset_df["sampling_strategy"], categories=subset_order, ordered=True
            )
            plot_df = (
                subset_df
                .groupby(["method", "sampling_strategy"], observed=True, as_index=False)
                .agg(
                    identified=("f1", lambda x: (x == 1.0).sum()),
                    total=("f1", "count")
                )
            )
            plot_df["identified"] = (plot_df["identified"] / plot_df["total"]) * 100
            strategy_positions = {strategy: position for position, strategy in enumerate(subset_order)}
            plot_df["x_pos"] = plot_df["sampling_strategy"].map(strategy_positions)

            sns.scatterplot(
                data=plot_df,
                x="x_pos",
                y="identified",
                hue="method",
                style="method",
                hue_order=method_names,
                style_order=method_names,
                markers=method_markers,
                palette=color_palette,
                alpha=0.7,
                s=40,
                ax=ax,
                linewidth=MARKERLINEWIDTH,
            )
            set_scatter_edgecolors(ax)

            for method_index, method in enumerate(method_names):
                method_points = plot_df[plot_df["method"] == method].sort_values("x_pos")
                if len(method_points) > 0:
                    ax.plot(
                        method_points["x_pos"],
                        method_points["identified"],
                        color=color_palette[method_index],
                        linestyle="--",
                        linewidth=1.2,
                        alpha=0.5,
                    )

            if legend_handles is None:
                legend_handles, legend_labels = ax.get_legend_handles_labels()

            ax.set_yticks([0, 50, 100])
            ax.set_ylim(0, 110)
            ax.tick_params(axis="x", labelsize=TICKFONTSIZE-1, bottom=True, width=0.5, length=3)
            ax.tick_params(axis="y", labelsize=TICKFONTSIZE, left=True, width=0.5, length=3)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color("black")
            ax.spines["bottom"].set_color("black")
            ax.grid(visible=True, which="major", linestyle="--", linewidth=0.5, alpha=0.7)

            ax.set_xticks(list(strategy_positions.values()))
            if subset_name == "random":
                tick_labels = [f"${strategy_labels[strategy]}|\\CMcal{{O}}|$" for strategy in subset_order]
            else:
                tick_labels = [strategy_labels[strategy] for strategy in subset_order]
            ax.set_xticklabels(tick_labels, rotation=0, ha="center")
            if len(subset_order) > 1:
                ax.set_xlim(-0.25, len(subset_order) - 0.75)
            if row_index == len(available_systems) - 1:
                ax.set_xlabel(
                    "$t$" if subset_name == "t-wise" else "\\# Configurations",
                    fontsize=LABELFONTSIZE-1,
                    labelpad=4,
                )
            else:
                ax.set_xlabel("")

            if col_index == 0:
                ax.set_ylabel(casestudy, fontsize=LEGENDFONTSIZE-1, labelpad=4)
                ax.yaxis.label.set_bbox(
                    dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.2", linewidth=0.3)
                )
            else:
                ax.set_ylabel("")

            if row_index == 0:
                ax.set_title(
                    "$t$-wise" if subset_name == "t-wise" else "random",
                    fontsize=LEGENDFONTSIZE,
                    pad=8,
                    bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.2", linewidth=0.3),
                )
            if ax.get_legend() is not None:
                ax.get_legend().remove()

        full_ax = axes[row_index][2]
        if not has_full:
            full_ax.set_visible(False)
            continue

        casestudy_df = df[df["casestudy"] == casestudy]
        full_df = casestudy_df[casestudy_df["sampling_strategy"] == "full"].copy()
        if full_df.empty:
            full_ax.set_visible(False)
            continue

        full_plot_df = (
            full_df
            .groupby(["method", "sampling_strategy"], observed=True, as_index=False)
            .agg(
                identified=("f1", lambda x: (x == 1.0).sum()),
                total=("f1", "count")
            )
        )
        full_plot_df["identified"] = (full_plot_df["identified"] / full_plot_df["total"]) * 100
        full_plot_df["method"] = pd.Categorical(full_plot_df["method"], categories=method_names, ordered=True)
        full_plot_df = full_plot_df.sort_values("method")
        x_positions = np.arange(len(method_names))

        full_ax.bar(
            x_positions,
            full_plot_df["identified"].to_numpy(),
            width=0.5,
            color=color_palette,
            edgecolor=get_edge_colors(color_palette),
            linewidth=MARKERLINEWIDTH,
        )
        # Set facecolor alpha while keeping edgecolor at full opacity
        for patch, color in zip(full_ax.patches, color_palette):
            patch.set_facecolor((*color, 0.5))
            patch.set_edgecolor(darken_color(color))

        if legend_handles is None:
            legend_handles, legend_labels = full_ax.get_legend_handles_labels()

        full_ax.set_yticks([0, 50, 100])
        full_ax.set_ylim(0, 105)
        full_ax.set_xticks([])
        full_ax.set_xlabel("")
        full_ax.tick_params(axis="x", bottom=False)
        full_ax.tick_params(axis="y", labelsize=TICKFONTSIZE, left=True, width=0.5, length=3)
        full_ax.spines["top"].set_visible(False)
        full_ax.spines["right"].set_visible(False)
        full_ax.spines["left"].set_color("black")
        full_ax.spines["bottom"].set_color("black")
        full_ax.grid(visible=True, which="major", axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
        full_ax.set_ylabel("")

        if row_index == 0:
            full_ax.set_title(
                "$\\CMcal{C}$",
                fontsize=LEGENDFONTSIZE,
                pad=8,
                bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.2", linewidth=0.3),
            )

        if full_ax.get_legend() is not None:
            full_ax.get_legend().remove()

    # Add average row
    avg_row_index = len(available_systems)
    for col_index, (subset_name, subset_order, axis_label) in enumerate(subsets):
        ax = axes[avg_row_index][col_index]
        # Aggregate across ALL case studies (not just displayed ones)
        subset_df = df_all[df_all["sampling_strategy"].isin(subset_order)].copy()
        if subset_df.empty or not subset_order:
            ax.set_visible(False)
            continue

        subset_df["sampling_strategy"] = pd.Categorical(
            subset_df["sampling_strategy"], categories=subset_order, ordered=True
        )
        plot_df = (
            subset_df
            .groupby(["method", "sampling_strategy"], observed=True, as_index=False)
            .agg(
                identified=("f1", lambda x: (x == 1.0).sum()),
                total=("f1", "count")
            )
        )
        plot_df["identified"] = (plot_df["identified"] / plot_df["total"]) * 100
        strategy_positions = {strategy: position for position, strategy in enumerate(subset_order)}
        plot_df["x_pos"] = plot_df["sampling_strategy"].map(strategy_positions)

        sns.scatterplot(
            data=plot_df,
            x="x_pos",
            y="identified",
            hue="method",
            style="method",
            hue_order=method_names,
            style_order=method_names,
            markers=method_markers,
            palette=color_palette,
            alpha=0.7,
            s=40,
            ax=ax,
            linewidth=MARKERLINEWIDTH,
        )
        set_scatter_edgecolors(ax)

        for method_index, method in enumerate(method_names):
            method_points = plot_df[plot_df["method"] == method].sort_values("x_pos")
            if len(method_points) > 0:
                ax.plot(
                    method_points["x_pos"],
                    method_points["identified"],
                    color=color_palette[method_index],
                    linestyle="--",
                    linewidth=1.2,
                    alpha=0.5,
                )

        # Capture legend handles if not yet captured
        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

        ax.set_yticks([0, 50, 100])
        ax.set_ylim(0, 110)
        ax.tick_params(axis="x", labelsize=TICKFONTSIZE-1, bottom=True, width=0.5, length=3)
        ax.tick_params(axis="y", labelsize=TICKFONTSIZE, left=True, width=0.5, length=3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("black")
        ax.spines["bottom"].set_color("black")
        ax.grid(visible=True, which="major", linestyle="--", linewidth=0.5, alpha=0.7)

        ax.set_xticks(list(strategy_positions.values()))
        if subset_name == "random":
            tick_labels = [f"${strategy_labels[strategy]}|\\CMcal{{O}}|$" for strategy in subset_order]
        else:
            tick_labels = [strategy_labels[strategy] for strategy in subset_order]
        ax.set_xticklabels(tick_labels, rotation=0, ha="center")
        if len(subset_order) > 1:
            ax.set_xlim(-0.25, len(subset_order) - 0.75)
        ax.set_xlabel(
            "$t$" if subset_name == "t-wise" else "\\# Configurations",
            fontsize=LABELFONTSIZE-1,
            labelpad=4,
        )

        if col_index == 0:
            ax.set_ylabel("\\textit{Average}", fontsize=LEGENDFONTSIZE-1, labelpad=4)
            ax.yaxis.label.set_bbox(
                dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.2", linewidth=0.3)
            )
        else:
            ax.set_ylabel("")

        # Add titles if this is the first row (happens when no individual systems are shown)
        if avg_row_index == 0:
            ax.set_title(
                "$t$-wise" if subset_name == "t-wise" else "random",
                fontsize=LEGENDFONTSIZE,
                pad=8,
                bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.2", linewidth=0.3),
            )

        if ax.get_legend() is not None:
            ax.get_legend().remove()

    # Handle full column for average row
    full_ax = axes[avg_row_index][2]
    if not has_full:
        full_ax.set_visible(False)
    else:
        # Use ALL case studies for average
        full_df = df_all[df_all["sampling_strategy"] == "full"].copy()
        if full_df.empty:
            full_ax.set_visible(False)
        else:
            full_plot_df = (
                full_df
                .groupby(["method", "sampling_strategy"], observed=True, as_index=False)
                .agg(
                    identified=("f1", lambda x: (x == 1.0).sum()),
                    total=("f1", "count")
                )
            )
            full_plot_df["identified"] = (full_plot_df["identified"] / full_plot_df["total"]) * 100
            full_plot_df["method"] = pd.Categorical(full_plot_df["method"], categories=method_names, ordered=True)
            full_plot_df = full_plot_df.sort_values("method")
            x_positions = np.arange(len(method_names))

            full_ax.bar(
                x_positions,
                full_plot_df["identified"].to_numpy(),
                width=0.7,
                color=color_palette,
                edgecolor=get_edge_colors(color_palette),
                linewidth=MARKERLINEWIDTH,
            )
            # Set facecolor alpha while keeping edgecolor at full opacity
            for patch, color in zip(full_ax.patches, color_palette):
                patch.set_facecolor((*color, 0.5))
                patch.set_edgecolor(darken_color(color))

            full_ax.set_yticks([0, 50, 100])
            full_ax.set_ylim(0, 105)
            full_ax.set_xticks([])
            full_ax.set_xlabel("")
            full_ax.tick_params(axis="x", bottom=False)
            full_ax.tick_params(axis="y", labelsize=TICKFONTSIZE, left=True, width=0.5, length=3)
            full_ax.spines["top"].set_visible(False)
            full_ax.spines["right"].set_visible(False)
            full_ax.spines["left"].set_color("black")
            full_ax.spines["bottom"].set_color("black")
            full_ax.grid(visible=True, which="major", axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
            full_ax.set_ylabel("")

            # Add title if this is the first row (happens when no individual systems are shown)
            if avg_row_index == 0:
                full_ax.set_title(
                    "$\\CMcal{C}$",
                    fontsize=LEGENDFONTSIZE,
                    pad=8,
                    bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.2", linewidth=0.3),
                )

            if full_ax.get_legend() is not None:
                full_ax.get_legend().remove()

    if legend_handles is not None and legend_labels is not None:
        # Update legend handle edge colors to full opacity
        for handle, color in zip(legend_handles, color_palette):
            if hasattr(handle, 'set_edgecolor'):
                handle.set_edgecolor(darken_color(color))
            if hasattr(handle, 'set_markeredgecolor'):
                handle.set_markeredgecolor(darken_color(color))
        fig.legend(
            legend_handles,
            legend_labels,
            title="",
            fontsize=LEGENDFONTSIZE-1,
            title_fontsize=LEGENDTITLEFONTSIZE,
            loc="upper center",
            ncol=len(method_names),
            bbox_to_anchor=(0.5, 1.03),
            edgecolor="white",
            handletextpad=0.2,
            columnspacing=0.7,
            handlelength=0.7,
        )

    fig.text(
        0.04, 0.5, "\\% Identified", ha="center", va="center", rotation=90,
        fontsize=LABELFONTSIZE, color="black"
    )
    plt.subplots_adjust(top=0.95, bottom=0.11, left=0.16, right=0.99, hspace=0.25, wspace=0.12)
    plt.savefig(get_output_path("rq2_sampling.pdf"), dpi=300, bbox_inches="tight", format="pdf")
    plt.close(fig)

@cli.command()
@click.option("--with-cart", is_flag=True, default=False, help="Include CART and write output to figures/out/cart.")
def plot_sampling_avg(with_cart):
    """Plot average sampling across all case studies with legend on the right."""
    global methods, method_names
    configure_plot_settings(with_cart)
    df_all = load_experiment_results_multi_method("sampling/distance_based", methods)
    df_all["method"] = pd.Categorical(df_all["method"], categories=method_names, ordered=True)
    df_all["sampling_strategy"] = df_all["sampling_strategy"].astype(str)
    df_all["sample_size"] = df_all["sample_size"].astype(int)

    all_strategies = set(df_all["sampling_strategy"].unique())
    twise_order = sorted(
        [strategy for strategy in all_strategies if strategy.endswith("-wise")],
        key=lambda strategy: int(strategy.split("-")[0]) if strategy.split("-")[0].isdigit() else 0,
    )
    random_order = sorted(
        [strategy for strategy in all_strategies if strategy.startswith("random_")],
        key=lambda strategy: int(strategy.split("_", 1)[1]) if strategy.split("_", 1)[1].isdigit() else 0,
    )

    strategy_labels = {
        strategy: (
            "$\\CMcal{C}$"
            if strategy == "full"
            else strategy.split("_", 1)[1]
            if strategy.startswith("random_")
            else strategy.split("-", 1)[0]
        )
        for strategy in all_strategies
    }

    subsets: List[tuple[str, List[str], str]] = [
        ("t-wise", twise_order, "Coverage"),
        ("random", random_order, "Sample Size"),
    ]

    n_cols = 3
    fig, axes = plt.subplots(
        1,
        n_cols,
        figsize=(6, 1.35),
        sharey=True,
        squeeze=False,
        gridspec_kw={"width_ratios": [1.0, 1.0, 0.3]},
    )

    color_palette = sns.color_palette(COLOR_PALETTE, len(method_names))
    method_markers = get_method_markers(method_names)
    legend_handles = None
    legend_labels = None

    for col_index, (subset_name, subset_order, axis_label) in enumerate(subsets):
        ax = axes[0][col_index]
        # Aggregate across ALL case studies
        subset_df = df_all[df_all["sampling_strategy"].isin(subset_order)].copy()
        if subset_df.empty or not subset_order:
            ax.set_visible(False)
            continue

        subset_df["sampling_strategy"] = pd.Categorical(
            subset_df["sampling_strategy"], categories=subset_order, ordered=True
        )
        plot_df = (
            subset_df
            .groupby(["method", "sampling_strategy"], observed=True, as_index=False)
            .agg(
                identified=("f1", lambda x: (x == 1.0).sum()),
                total=("f1", "count")
            )
        )
        plot_df["identified"] = (plot_df["identified"] / plot_df["total"]) * 100
        strategy_positions = {strategy: position for position, strategy in enumerate(subset_order)}
        plot_df["x_pos"] = plot_df["sampling_strategy"].map(strategy_positions)

        sns.scatterplot(
            data=plot_df,
            x="x_pos",
            y="identified",
            hue="method",
            style="method",
            hue_order=method_names,
            style_order=method_names,
            markers=method_markers,
            palette=color_palette,
            alpha=0.7,
            s=40,
            ax=ax,
            linewidth=MARKERLINEWIDTH,
        )
        set_scatter_edgecolors(ax)

        for method_index, method in enumerate(method_names):
            method_points = plot_df[plot_df["method"] == method].sort_values("x_pos")
            if len(method_points) > 0:
                ax.plot(
                    method_points["x_pos"],
                    method_points["identified"],
                    color=color_palette[method_index],
                    linestyle="--",
                    linewidth=1.2,
                    alpha=0.5,
                )

        # Capture legend handles if not yet captured
        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

        ax.set_yticks([0, 50, 100])
        ax.set_ylim(0, 110)
        ax.tick_params(axis="x", labelsize=TICKFONTSIZE-1, bottom=True, width=0.5, length=3)
        ax.tick_params(axis="y", labelsize=TICKFONTSIZE, left=True, width=0.5, length=3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("black")
        ax.spines["bottom"].set_color("black")
        ax.grid(visible=True, which="major", linestyle="--", linewidth=0.5, alpha=0.7)

        ax.set_xticks(list(strategy_positions.values()))
        if subset_name == "random":
            tick_labels = [f"${strategy_labels[strategy]}|\\CMcal{{O}}|$" for strategy in subset_order]
        else:
            tick_labels = [strategy_labels[strategy] for strategy in subset_order]
        ax.set_xticklabels(tick_labels, rotation=0, ha="center")
        if len(subset_order) > 1:
            ax.set_xlim(-0.25, len(subset_order) - 0.75)
        ax.set_xlabel(
            "$t$" if subset_name == "t-wise" else "\\# Configurations",
            fontsize=LABELFONTSIZE-1,
            labelpad=4,
        )

        if col_index == 0:
            ax.set_ylabel("\\% Identified", fontsize=LABELFONTSIZE-1, labelpad=4)
        else:
            ax.set_ylabel("")

        ax.set_title(
            "$t$-wise" if subset_name == "t-wise" else "random",
            fontsize=LEGENDFONTSIZE,
            pad=YLABELPAD
        )

        if ax.get_legend() is not None:
            ax.get_legend().remove()

    # Handle full column
    full_ax = axes[0][2]
    full_df = df_all[df_all["sampling_strategy"] == "full"].copy()
    if full_df.empty:
        full_ax.set_visible(False)
    else:
        full_plot_df = (
            full_df
            .groupby(["method", "sampling_strategy"], observed=True, as_index=False)
            .agg(
                identified=("f1", lambda x: (x == 1.0).sum()),
                total=("f1", "count")
            )
        )
        full_plot_df["identified"] = (full_plot_df["identified"] / full_plot_df["total"]) * 100
        full_plot_df["method"] = pd.Categorical(full_plot_df["method"], categories=method_names, ordered=True)
        full_plot_df = full_plot_df.sort_values("method")
        x_positions = np.arange(len(method_names))

        full_ax.bar(
            x_positions,
            full_plot_df["identified"].to_numpy(),
            width=0.7,
            color=color_palette,
            edgecolor=get_edge_colors(color_palette),
            linewidth=MARKERLINEWIDTH,
        )
        # Set facecolor alpha while keeping edgecolor at full opacity
        for patch, color in zip(full_ax.patches, color_palette):
            patch.set_facecolor((*color, 0.5))
            patch.set_edgecolor(darken_color(color))

        full_ax.set_yticks([0, 50, 100])
        full_ax.set_ylim(0, 105)
        full_ax.set_xticks([])
        full_ax.set_xlabel("")
        full_ax.tick_params(axis="x", bottom=False)
        full_ax.tick_params(axis="y", labelsize=TICKFONTSIZE, left=True, width=0.5, length=3)
        full_ax.spines["top"].set_visible(False)
        full_ax.spines["right"].set_visible(False)
        full_ax.spines["left"].set_color("black")
        full_ax.spines["bottom"].set_color("black")
        full_ax.grid(visible=True, which="major", axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
        full_ax.set_ylabel("")

        full_ax.set_title(
            "$\\CMcal{C}$",
            fontsize=LEGENDFONTSIZE,
            pad=YLABELPAD
        )

        if full_ax.get_legend() is not None:
            full_ax.get_legend().remove()

    # Add legend on the right
    if legend_handles is not None and legend_labels is not None:
        # Create custom legend handles with method names
        from matplotlib.lines import Line2D
        legend_handles_custom = []
        for j, method in enumerate(method_names):
            legend_handles_custom.append(Line2D([0], [0], marker=method_markers[j], color=color_palette[j], 
                                        linestyle='--', linewidth=1.2, markersize=6,
                                        markerfacecolor=(*color_palette[j], 0.7),
                                        markeredgecolor=darken_color(color_palette[j]), markeredgewidth=0.5))
        
        # Place legend to the right of the rightmost axis
        rightmost_ax = axes[0][n_cols - 1]
        rightmost_ax.legend(
            legend_handles_custom,
            method_names,
            fontsize=LEGENDFONTSIZE-1,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            edgecolor="white",
            handletextpad=0.3,
            frameon=True,
        )

    plt.subplots_adjust(left=0.08, right=0.77, top=0.92, bottom=0.18, wspace=0.1)
    plt.savefig(get_output_path("rq2_sampling_avg.pdf"), dpi=300, format="pdf", bbox_inches="tight")
    plt.close(fig)

def statistical_significance_trend(experiment_path="scalability_features/workload", target_column="n_features"):
    """
    For each method, test for a monotonic trend (e.g., decreasing F1) across ordered groups
    using the Mann-Kendall test (nonparametric test for monotonic trend).
    """
    df = load_experiment_results_multi_method(experiment_path, ["rsd", "beam-kl", "beam-mean", "dfs-mean", "syflow"])
    methods = df["method"].unique()
    results = {}

    for method in methods:
        method_df = df[df["method"] == method]
        sorted_df = method_df.sort_values(by=target_column)
        sorted_df["identified"] = sorted_df.groupby([target_column])["f1"].transform(lambda x: (x == 1.0).sum())
        grouped = sorted_df.groupby(target_column).agg({"identified": "max"}).reset_index()
        mk_result = mk.original_test(grouped["identified"], alpha=0.05)

        results[method] = mk_result
        print(f"Method: {method}")
        print(mk_result)
        print("-" * 40)

    return results

@cli.command()
def significance_trend_features():
    """Test for monotonic trend in F1 scores across number of features."""
    statistical_significance_trend("scalability_features/workload", "n_features")

@cli.command()
def significance_trend_samples():
    """Test for monotonic trend in F1 scores across number of samples."""
    statistical_significance_trend("scalability_samples/workload", "n_samples")

if __name__ == "__main__":
    cli()