import click
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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
    LEGENDTITLEFONTSIZE,
    XLABELPAD,
    YLABELPAD,
    MARKERLINEWIDTH,
    init_plot_style,
    load_experiment_results,
    darken_color,
    set_scatter_edgecolors
)
from stats_utils import (
    compare_methods,
    friedman_per_method,
    identified_rate,
    trend_per_method,
    write_stats_table,
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


def configure_plot_settings(full_mode=False):
    global methods, method_names, output_dir
    methods = CART_METHODS.copy() if full_mode else DEFAULT_METHODS.copy()
    method_names = CART_METHOD_NAMES.copy() if full_mode else DEFAULT_METHOD_NAMES.copy()
    output_dir = Path("figures/out/full") if full_mode else Path("figures/out")


def get_method_markers(active_method_names):
    return [METHOD_MARKERS[method_name] for method_name in active_method_names]


def get_output_path(filename):
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / filename


def get_subplot_grid(n_panels):
    n_cols = 3 if n_panels > 4 else 2
    n_rows = int(np.ceil(n_panels / n_cols))
    return n_rows, n_cols

def load_experiment_results_multi_method(experiment_path, methods = ["syflow", "rsd", "beam-kl", "beam-mean", "dfs-mean"]):
    frames = []
    for method in methods:
        folder = f"results/RQ1/{method}/{experiment_path}"
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
    configure_plot_settings(full_mode=False)

@cli.command()
@click.option("--full", "full_mode", is_flag=True, default=False, help="Enable full mode: include CART and plot all subject systems.")
def plot_number_of_subgroups(full_mode):
    """Plot results for varying the number of subgroups."""
    global methods, method_names
    configure_plot_settings(full_mode)
    df_wl = load_experiment_results_multi_method("number_of_subgroups/workload", methods)
    df_db = load_experiment_results_multi_method("number_of_subgroups/distance_based", methods)
    df_all = pd.concat([df_wl, df_db])
    df_all["method"] = pd.Categorical(df_all["method"], categories=method_names, ordered=True)
    
    # Drop case studies that don't have 3 groups
    df_all = df_all[df_all["casestudy"].isin(df_all[df_all["n_groups"] == 3]["casestudy"].unique())]
    df_all["casestudy"] = df_all["casestudy"].astype("str")

    representative_systems = [
        "\\textsc{jump3r}",
        "\\textsc{kanzi}",
        "\\textsc{7z}",
        "\\textsc{BDB-C}",
        "\\textsc{Polly}",
    ]

    if full_mode:
        # In full mode, show all available systems across methods.
        # Methods with missing systems are simply absent for those boxes.
        selected_case_studies = sorted(df_all["casestudy"].unique())
    else:
        selected_case_studies = [
            system for system in representative_systems if system in set(df_all["casestudy"].unique())
        ]

    legend_case_studies = selected_case_studies + ["\\textit{Average}"]
    legend_entries = len(legend_case_studies)
    legend_cols = min(legend_entries, 6)
    legend_rows = int(np.ceil(legend_entries / legend_cols))

    if full_mode:
        fig_width = max(8.5, 5.5 + 0.45 * legend_entries)
    else:
        fig_width = 6.5
    fig_height = 0.8 * len(method_names) + 0.25 * max(0, legend_rows - 1)

    # Create a figure with subplots for each method
    fig, axes = plt.subplots(len(method_names), 1, figsize=(fig_width, fig_height), sharex=True, sharey=True)

    for i, method in enumerate(method_names):
        ax = axes[i] if len(method_names) > 1 else axes
        # Get ALL case studies for this method (for average computation)
        method_df_all = df_all[df_all["method"] == method].copy()
        method_df_all["mean_f1"] = method_df_all[["f1_0", "f1_1", "f1_2"]].sum(axis=1) / method_df_all["n_groups"]
        
        method_df = method_df_all[method_df_all["casestudy"].isin(selected_case_studies)].copy()

        # Add average data by pooling ALL case studies for this method
        avg_method_df = method_df_all.copy()
        avg_method_df["casestudy"] = "\\textit{Average}"
        
        # Combine with original data
        method_df_plot = pd.concat([method_df, avg_method_df], ignore_index=True)
        method_df_plot["casestudy"] = pd.Categorical(
            method_df_plot["casestudy"], categories=legend_case_studies, ordered=True
        )

        box_palette = sns.color_palette(COLOR_PALETTE, len(method_df_plot["casestudy"].unique()))
        sns.boxplot(
            data=method_df_plot,
            x="n_groups",
            y="mean_f1",
            hue="casestudy",
            hue_order=legend_case_studies,
            palette=COLOR_PALETTE,
            ax=ax,
            gap=0.4,
            width=0.85,
            linewidth=MARKERLINEWIDTH,
            flierprops=dict(marker="o", markersize=3, markeredgewidth=0.5),
            boxprops=dict(edgecolor="black"),
        )
        
        # Calculate actual number of boxes (x positions * hue levels)
        n_x = len(method_df_plot["n_groups"].unique())
        n_hue = len(legend_case_studies)
        n_actual_boxes = n_x * n_hue
        
        # Store the dark colors and face colors for each box based on its face color (only actual boxes, not legend)
        box_dark_colors = []
        box_face_colors = []
        for i, patch in enumerate(ax.patches[:n_actual_boxes]):
            if hasattr(patch, 'get_facecolor'):
                fc = patch.get_facecolor()
                dark_color = darken_color(fc[:3])
                # Set facecolor with alpha, but edgecolor at full opacity
                patch.set_facecolor((*fc[:3], 0.7))
                patch.set_edgecolor(dark_color)
                box_dark_colors.append(dark_color)
                box_face_colors.append(fc[:3])
        
        # Color whiskers, caps, medians, and fliers to match their boxes
        # Each box has 6 lines: 2 whiskers, 2 caps, 1 median, 1 flier line
        # Lines are organized: all lines for box 0, then all lines for box 1, etc.
        lines_per_box = 6
        for idx, line in enumerate(ax.lines):
            box_idx = idx // lines_per_box
            if box_idx < len(box_dark_colors):
                line.set_color(box_dark_colors[box_idx])
                # Fliers have marker='o' - color their markers too
                if line.get_marker() == 'o':
                    line.set_markerfacecolor('none')  # no fill, just outline
                    line.set_markeredgecolor(box_dark_colors[box_idx])
        
        ax.set_yticks([0, 0.333, 0.666, 1.0])
        ax.set_yticklabels([0, "", "", 1])
        
        ax.set_xlabel(r"\# Seeded Subspaces", fontsize=LABELFONTSIZE, labelpad=XLABELPAD)
        ax.set_ylabel(method, fontsize=LABELFONTSIZE-4, labelpad=YLABELPAD)
        ax.tick_params(axis="x", labelsize=TICKFONTSIZE, bottom=True, width=0.5, length=3)
        ax.tick_params(axis="y", labelsize=TICKFONTSIZE, left=True, width=0.5, length=3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("black")
        ax.spines["bottom"].set_color("black")
        ax.grid(visible=True, which="major", linestyle="--", linewidth=0.5, alpha=0.7)

        n_conditions_sorted = sorted(method_df_plot["n_groups"].unique())
        # Use the order from the boxplot legend to match hue positions
        handles, labels = ax.get_legend_handles_labels()
        casestudies_sorted = labels
        n_casestudies = len(casestudies_sorted)

        # Calculate positions matching seaborn's boxplot with gap=0.4, width=0.8
        # Seaborn distributes boxes within each x position
        total_width = 0.85
        
        for j, n_groups in enumerate(n_conditions_sorted):
            # Map n_groups value to x position index
            x_base = n_conditions_sorted.index(n_groups)
            
            for k, casestudy in enumerate(casestudies_sorted):
                if casestudy == "\\textit{Average}":
                    # For average, compute the mean count across ALL case studies (not just displayed ones)
                    all_casestudies = method_df_all["casestudy"].unique()
                    count_at_cap = sum(
                        (method_df_all[(method_df_all["n_groups"] == n_groups) & (method_df_all["casestudy"] == cs)]["mean_f1"] == 1.0).sum()
                        for cs in all_casestudies
                    ) / len(all_casestudies)
                    # Always display average
                    offset = (k - (n_casestudies - 1) / 2) * (total_width / n_casestudies)
                    x_pos = x_base + offset
                    ax.text(
                        x_pos, 1.0 + 0.01, f"\\textbf{{{count_at_cap:.0f}}}",
                        ha="center", va="bottom", fontsize=TICKFONTSIZE-1, color=sns.color_palette(COLOR_PALETTE, 15)[k]
                    )
                else:
                    subset = method_df[(method_df["n_groups"] == n_groups) & (method_df["casestudy"] == casestudy)]
                    # Only display if there's data for this combination
                    if len(subset) > 0:
                        count_at_cap = (subset["mean_f1"] == 1.0).sum()
                        offset = (k - (n_casestudies - 1) / 2) * (total_width / n_casestudies)
                        x_pos = x_base + offset
                        ax.text(
                            x_pos, 1.0 + 0.01, f"\\textbf{{{count_at_cap}}}",
                            ha="center", va="bottom", fontsize=TICKFONTSIZE-1, color=sns.color_palette(COLOR_PALETTE, 15)[k]
                        )
        ax.legend_.remove()

    # Add a single legend for all subplots
    handles, labels = ax.get_legend_handles_labels()
    # Update legend handle edge colors to match the plot styling
    palette = sns.color_palette(COLOR_PALETTE, len(labels))
    for handle, color in zip(handles, palette):
        handle.set_edgecolor(darken_color(color))
        handle.set_facecolor((color, 0.7))
    fig.legend(
        handles,
        labels,
        title="",
        fontsize=LEGENDFONTSIZE,
        title_fontsize=LEGENDTITLEFONTSIZE,
        loc="upper center",
        ncol=legend_cols,
        bbox_to_anchor=(0.5, 1.02 + 0.05 * (legend_rows - 1)),
        edgecolor="white",
        # decrease width of boxes in legend
        handletextpad=0.3,
        borderaxespad=0.3,
        columnspacing=1.4,
        handlelength=0.7,
    )
    fig.text(
        0.035, 0.5, "Mean F1 Score", ha="center", va="center", rotation=90,
        fontsize=LABELFONTSIZE, color="black"
    )
    plt.subplots_adjust(top=max(0.72, 0.90 - 0.08 * (legend_rows - 1)), hspace=0.55)
    plt.savefig(get_output_path("rq1_number_of_subgroups.pdf"), dpi=300, bbox_inches="tight", format="pdf")

@cli.command()
@click.option("--full", "full_mode", is_flag=True, default=False, help="Enable full mode: include CART and plot all subject systems.")
def plot_distributions(full_mode):
    """Plot performance across different distributions for full configuration set."""
    global methods, method_names
    configure_plot_settings(full_mode)
    df_db = load_experiment_results_multi_method("distributions/distance_based", methods)
    df_wl = load_experiment_results_multi_method("distributions/workload", methods)
    df_all = pd.concat([df_db, df_wl], ignore_index=True)
    df_all["method"] = pd.Categorical(df_all["method"], categories=method_names, ordered=True)
    
    # Map distribution names to mathematical symbols
    dist_symbols = {
        "normal": r"$\CMcal{N}$",
        "bi_modal_split": r"$\CMcal{N}_2$",
        "tri_modal_split": r"$\CMcal{N}_3$",
        "power_law": r"$\CMcal{P}$",
        "uniform": r"$\CMcal{U}$",
    }
    
    # Compute average from ALL case studies
    avg_plot_df_all = (
        df_all.groupby(["method", "sg_distribution", "casestudy"], observed=True, as_index=False)
        .agg(
            identified=("f1", lambda x: (x == 1.0).sum()),
            total=("f1", "count"),
        )
    )
    avg_plot_df_all["identified_pct"] = (avg_plot_df_all["identified"] / avg_plot_df_all["total"]) * 100
    avg_plot_df = avg_plot_df_all.groupby(["method", "sg_distribution"], observed=True, as_index=False).agg({
        "identified_pct": "mean"
    })
    
    # Now filter to representative systems for display
    representative_systems = [
        "\\textsc{LLVM}",
        "\\textsc{VP9}",
        "\\textsc{xz}",
    ]
    if full_mode:
        available_systems = sorted(df_all["casestudy"].unique())
    else:
        available_systems = [system for system in representative_systems if system in df_all["casestudy"].unique()]
    df = df_all[df_all["casestudy"].isin(available_systems)].copy()
    df["casestudy"] = pd.Categorical(df["casestudy"], categories=available_systems, ordered=True)
    
    # Calculate percentage identified for display case studies
    plot_df = (
        df.groupby(["method", "sg_distribution", "casestudy"], observed=True, as_index=False)
        .agg(
            identified=("f1", lambda x: (x == 1.0).sum()),
            total=("f1", "count"),
        )
    )
    plot_df["identified_pct"] = (plot_df["identified"] / plot_df["total"]) * 100
    
    # Define logical distribution order
    distribution_order = ["normal", "bi_modal_split", "tri_modal_split", "uniform", "power_law"]
    all_distributions = [d for d in distribution_order if d in plot_df["sg_distribution"].unique()]
    
    # Create subplots (one heatmap per system + one for average) and wrap them into rows in full mode.
    n_systems = len(available_systems)
    n_methods = len(method_names)
    n_dists = len(all_distributions)
    n_panels = n_systems + 1
    if full_mode:
        n_cols = min(4, n_panels)
    else:
        n_cols = n_panels
    n_rows = int(np.ceil(n_panels / n_cols))

    panel_width = 1.3
    panel_height = 1.5
    fig_width = max(5.8, n_cols * panel_width + 0.9)
    fig_height = max(2.2, n_rows * panel_height + 0.6)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), sharey=False)
    axes = np.atleast_1d(axes).flatten()

    # Use a color gradient from the muted palette for consistency
    muted_palette = sns.color_palette(COLOR_PALETTE)
    # Create a sequential colormap with more color stops concentrated at higher percentages
    # This provides better distinction at 70-100% while keeping ticks equidistant
    dist_cmap = mcolors.LinearSegmentedColormap.from_list(
        "muted_sequential",
        [
            (0, "white"),
            (0.8, tuple(min(1.0, 1.1 * c) for c in mcolors.to_rgb(muted_palette[0]))), 
            (1.0, tuple(0.8 * c for c in mcolors.to_rgb(muted_palette[0]))),
        ],
    )
    # Use linear normalization to keep colorbar ticks equidistant
    dist_norm = mcolors.Normalize(vmin=0, vmax=100)
    
    def plot_distribution_heatmap(ax, heatmap_data, title, show_ylabels):
        heatmap = sns.heatmap(
            heatmap_data,
            annot=False,
            cmap=dist_cmap,
            norm=dist_norm,
            cbar=False,
            linewidths=0.5,
            linecolor="white",
            ax=ax,
            square=True,
            xticklabels=True,
            yticklabels=True,
        )

        col_names = heatmap_data.columns.tolist()
        xtick_labels = [dist_symbols.get(dist, dist) for dist in col_names]
        ax.set_yticklabels(method_names if show_ylabels else [], fontsize=TICKFONTSIZE, rotation=0)
        ax.set_xticklabels(xtick_labels, fontsize=TICKFONTSIZE-2, rotation=0, ha='center')
        ax.tick_params(axis='x', pad=0)
        ax.tick_params(axis='y', pad=0)
        ax.set_xlabel("")
        ax.set_ylabel("Method" if show_ylabels else "", fontsize=LABELFONTSIZE, labelpad=YLABELPAD)
        ax.set_title(title, fontsize=LABELFONTSIZE, pad=4)
        return heatmap

    heatmap_obj = None
    for panel_idx, casestudy in enumerate(available_systems):
        ax = axes[panel_idx]
        casestudy_df = plot_df[plot_df["casestudy"] == casestudy]
        
        # Pivot to create heatmap data: rows=methods, cols=distributions (swapped)
        heatmap_data = casestudy_df.pivot(
            index="method",
            columns="sg_distribution",
            values="identified_pct"
        )
        # Reorder to match method_names and all_distributions
        heatmap_data = heatmap_data.reindex(index=method_names, columns=all_distributions)
        
        heatmap_obj = plot_distribution_heatmap(
            ax,
            heatmap_data,
            casestudy,
            show_ylabels=(panel_idx % n_cols == 0),
        )
    
    # Add average heatmap across all systems (using pre-computed average from ALL case studies)
    ax_avg = axes[n_systems]
    avg_heatmap_data = avg_plot_df.pivot(
        index="method",
        columns="sg_distribution",
        values="identified_pct"
    )
    avg_heatmap_data = avg_heatmap_data.reindex(index=method_names, columns=all_distributions)
    heatmap_obj = plot_distribution_heatmap(
        ax_avg,
        avg_heatmap_data,
        "\\textit{Average}",
        show_ylabels=(n_systems % n_cols == 0),
    )

    for empty_ax in axes[n_panels:]:
        empty_ax.set_visible(False)
    
    # Add a single colorbar to the right of all plots
    if heatmap_obj is not None:
        cbar_ax = fig.add_axes([0.905, 0.18, 0.015, 0.68])
        cbar = fig.colorbar(heatmap_obj.collections[0], cax=cbar_ax)
        cbar.set_label("\\% Identified", fontsize=LABELFONTSIZE)
        # Set equidistant ticks at regular intervals
        cbar.set_ticks([0, 20, 40, 60, 80, 100])
        cbar.ax.tick_params(length=3, width=0.5, labelsize=TICKFONTSIZE-2, right=True)
        cbar.outline.set_visible(False)
        # Add right spine to colorbar
        cbar.ax.spines['right'].set_visible(True)
        cbar.ax.spines['right'].set_color('black')
        cbar.ax.spines['right'].set_linewidth(0.5)
    
    plt.subplots_adjust(left=0.08, right=0.88, top=0.92, bottom=0.14, wspace=0.08, hspace=0.55)
    plt.savefig(get_output_path("rq1_distributions.pdf"), dpi=300, bbox_inches="tight", format="pdf")

@cli.command()
@click.option("--full", "full_mode", is_flag=True, default=False, help="Enable full mode: include CART and plot all subject systems.")
def plot_number_of_predicates(full_mode):
    '''Plot results for varying the number of predicates.'''
    global methods, method_names
    configure_plot_settings(full_mode)
    df_wl = load_experiment_results_multi_method("number_of_predicates/workload", methods)
    df_db = load_experiment_results_multi_method("number_of_predicates/distance_based", methods)
    df_all = pd.concat([df_wl, df_db])
    df_all["method"] = pd.Categorical(df_all["method"], categories=method_names, ordered=True)
    
    # Compute average from ALL case studies
    df_scatter_all = df_all[["seed", "casestudy", "method", "n_conditions", "f1"]].copy()
    df_scatter_all["identified"] = df_all.groupby(["method", "n_conditions", "casestudy"])["f1"].transform(lambda x: (x == 1.0).sum())    
    df_scatter_all = df_scatter_all.groupby(["method", "n_conditions", "casestudy"]).agg({"identified": "max"}).reset_index()
    avg_df = df_scatter_all.groupby(["method", "n_conditions"], as_index=False).agg({"identified": "mean"})
    
    representative_systems = ["\\textsc{7z}", "\\textsc{BDB-C}", "\\textsc{h2}", "\\textsc{jump3r}", "\\textsc{z3}"]
    if full_mode:
        casestudies = sorted(df_all["casestudy"].unique())
    else:
        casestudies = [system for system in representative_systems if system in df_all["casestudy"].unique()]
    df = df_all[df_all["casestudy"].isin(casestudies)]
    color_palette = sns.color_palette(COLOR_PALETTE, len(method_names))
    method_markers = get_method_markers(method_names)

    df_scatter = df[["seed", "casestudy", "method", "n_conditions", "f1"]].copy()
    df_scatter["identified"] = df.groupby(["method", "n_conditions", "casestudy"])["f1"].transform(lambda x: (x == 1.0).sum())    
    df_scatter = df_scatter.groupby(["method", "n_conditions", "casestudy"]).agg({"identified": "max"}).reset_index()

    n_rows, n_cols = get_subplot_grid(len(casestudies) + 1)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.2, 1.15 * n_rows + 0.45), sharex=True, sharey=True)
    axes = axes.flatten()
    for i, casestudy in enumerate(casestudies):
        ax = axes[i]
        casestudy_df = df_scatter[df_scatter["casestudy"] == casestudy]
        sns.scatterplot(
            data=casestudy_df,
            x="n_conditions",
            y="identified",
            hue="method",
            style="method",
            markers=method_markers,
            palette=color_palette,
            alpha=0.7,
            s=40,
            ax=ax,
            linewidth=MARKERLINEWIDTH,
        )
        set_scatter_edgecolors(ax)
        # add dashed lines for each method
        for j, method in enumerate(method_names):#, "\\textsc{CART}"]):
            subset = casestudy_df[casestudy_df["method"] == method]
            if len(subset) > 0:
                ax.plot(subset["n_conditions"], subset["identified"], color=color_palette[j], linestyle="--", linewidth=1.2, alpha=0.5)

        ax.set_xlabel("", fontsize=LABELFONTSIZE, labelpad=XLABELPAD)
        ax.set_ylabel("", fontsize=LABELFONTSIZE, labelpad=YLABELPAD)
        # put casestudy name on the bottom left corner of the plot, surrounded by a box
        ax.text(
            1, 8, casestudy,
            ha="left", va="bottom", fontsize=LEGENDFONTSIZE,
            color="black", bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.2", linewidth=0.3)
        )
        ax.set_ylim(0, 105)
        ax.set_yticks([0, 50, 100])
        ax.set_xticks([1, 2, 3, 4, 5, 6, 7])
        ax.tick_params(axis="x", labelsize=TICKFONTSIZE, bottom=True, width=0.5, length=3)
        ax.tick_params(axis="y", labelsize=TICKFONTSIZE, left=True, width=0.5, length=3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("black")
        ax.spines["bottom"].set_color("black")
        ax.grid(visible=True, which="major", linestyle="--", linewidth=0.5, alpha=0.7)
        ax.get_legend().remove()

    # Add average subplot (using pre-computed average from ALL case studies)
    ax_avg = axes[len(casestudies)]
    
    sns.scatterplot(
        data=avg_df,
        x="n_conditions",
        y="identified",
        hue="method",
        style="method",
        markers=method_markers,
        palette=color_palette,
        alpha=0.7,
        s=40,
        ax=ax_avg,
        linewidth=MARKERLINEWIDTH,
    )
    set_scatter_edgecolors(ax_avg)
    # add dashed lines for each method
    for j, method in enumerate(method_names):
        subset = avg_df[avg_df["method"] == method]
        if len(subset) > 0:
            ax_avg.plot(subset["n_conditions"], subset["identified"], color=color_palette[j], linestyle="--", linewidth=1.2, alpha=0.5)

    ax_avg.set_xlabel("", fontsize=LABELFONTSIZE, labelpad=XLABELPAD)
    ax_avg.set_ylabel("", fontsize=LABELFONTSIZE, labelpad=YLABELPAD)
    # put "Average" label in the bottom left corner
    ax_avg.text(
        1, 8, "\\textit{Average}",
        ha="left", va="bottom", fontsize=LEGENDFONTSIZE,
        color="black", bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.2", linewidth=0.3)
    )
    ax_avg.set_ylim(0, 105)
    ax_avg.set_yticks([0, 50, 100])
    ax_avg.set_xticks([1, 2, 3, 4, 5, 6, 7])
    ax_avg.tick_params(axis="x", labelsize=TICKFONTSIZE, bottom=True, width=0.5, length=3)
    ax_avg.tick_params(axis="y", labelsize=TICKFONTSIZE, left=True, width=0.5, length=3)
    ax_avg.spines["top"].set_visible(False)
    ax_avg.spines["right"].set_visible(False)
    ax_avg.spines["left"].set_color("black")
    ax_avg.spines["bottom"].set_color("black")
    ax_avg.grid(visible=True, which="major", linestyle="--", linewidth=0.5, alpha=0.7)
    ax_avg.get_legend().remove()

    for empty_ax in axes[len(casestudies) + 1:]:
        empty_ax.set_visible(False)

    fig.text(
        0, 0.5, "\\% Identified", ha="center", va="center", rotation=90,
        fontsize=LABELFONTSIZE, color="black"
    )
    fig.text(
        0.5, 0, "\\# Predicates", ha="center", va="center",
        fontsize=LABELFONTSIZE, color="black"
    )
    # Add a single legend for all plots
    handles, labels = ax.get_legend_handles_labels()
    # Update legend handle edge colors to full opacity
    for handle, color in zip(handles, color_palette):
        if hasattr(handle, 'set_edgecolor'):
            handle.set_edgecolor(darken_color(color))
        if hasattr(handle, 'set_markeredgecolor'):
            handle.set_markeredgecolor(darken_color(color))
    fig.legend(
        handles,
        labels,
        title="",
        fontsize=LEGENDFONTSIZE,
        title_fontsize=LEGENDTITLEFONTSIZE,
        loc="upper center",
        ncol=len(method_names),
        bbox_to_anchor=(0.5, 1.05),
        edgecolor="white",
        handletextpad=0,
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.94, hspace=0.25, wspace=0.13)
    plt.savefig(get_output_path("rq1_number_of_predicates.pdf"), dpi=300, bbox_inches="tight", format="pdf")

@cli.command()
@click.option("--full", "full_mode", is_flag=True, default=False, help="Enable full mode: include CART and plot all subject systems.")
def plot_number_of_predicates_avg(full_mode):
    '''Plot average results for varying the number of predicates with legend on the right.'''
    global methods, method_names
    configure_plot_settings(full_mode)
    df_wl = load_experiment_results_multi_method("number_of_predicates/workload", methods)
    df_db = load_experiment_results_multi_method("number_of_predicates/distance_based", methods)
    df_all = pd.concat([df_wl, df_db])
    df_all["method"] = pd.Categorical(df_all["method"], categories=method_names, ordered=True)
    
    # Compute average from ALL case studies
    df_scatter_all = df_all[["seed", "casestudy", "method", "n_conditions", "f1"]].copy()
    df_scatter_all["identified"] = df_all.groupby(["method", "n_conditions", "casestudy"])["f1"].transform(lambda x: (x == 1.0).sum())    
    df_scatter_all = df_scatter_all.groupby(["method", "n_conditions", "casestudy"]).agg({"identified": "max"}).reset_index()
    avg_df = df_scatter_all.groupby(["method", "n_conditions"], as_index=False).agg({"identified": "mean"})
    
    color_palette = sns.color_palette(COLOR_PALETTE, len(method_names))
    method_markers = get_method_markers(method_names)

    fig, ax = plt.subplots(1, 1, figsize=(6, 1.8))
    
    sns.scatterplot(
        data=avg_df,
        x="n_conditions",
        y="identified",
        hue="method",
        style="method",
        markers=method_markers,
        palette=color_palette,
        alpha=0.7,
        s=40,
        ax=ax,
        linewidth=MARKERLINEWIDTH,
        legend=False,
    )
    set_scatter_edgecolors(ax)
    # add dashed lines for each method
    for j, method in enumerate(method_names):
        subset = avg_df[avg_df["method"] == method]
        if len(subset) > 0:
            ax.plot(subset["n_conditions"], subset["identified"], color=color_palette[j], linestyle="--", linewidth=1.2, alpha=0.5)

    ax.set_xlabel("\\# Predicates", fontsize=LABELFONTSIZE-1, labelpad=XLABELPAD)
    ax.set_ylabel("\\% Identified", fontsize=LABELFONTSIZE-1, labelpad=YLABELPAD)
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 50, 100])
    ax.set_xticks([1, 2, 3, 4, 5, 6, 7])
    ax.tick_params(axis="x", labelsize=TICKFONTSIZE, bottom=True, width=0.5, length=3)
    ax.tick_params(axis="y", labelsize=TICKFONTSIZE, left=True, width=0.5, length=3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.grid(visible=True, which="major", linestyle="--", linewidth=0.5, alpha=0.7)

    # Create custom legend handles with method names
    from matplotlib.lines import Line2D
    legend_handles = []
    for j, method in enumerate(method_names):
        legend_handles.append(Line2D([0], [0], marker=method_markers[j], color=color_palette[j], 
                                    linestyle='--', linewidth=1.2, markersize=6,
                                    markerfacecolor=(*color_palette[j], 0.7),
                                    markeredgecolor=darken_color(color_palette[j]), markeredgewidth=0.4))
    
    ax.legend(
        legend_handles,
        method_names,
        fontsize=LEGENDFONTSIZE-1,
        loc="center left",
        bbox_to_anchor=(1.02, 0.37),
        edgecolor="white",
        handletextpad=0.3,
        frameon=True,
    )

    plt.tight_layout()
    plt.savefig(get_output_path("rq1_number_of_predicates_avg.pdf"), dpi=300, bbox_inches="tight", format="pdf")
    plt.close(fig)

def load_number_of_subgroups_identified():
    df = pd.concat([
        load_experiment_results_multi_method("number_of_subgroups/workload", methods),
        load_experiment_results_multi_method("number_of_subgroups/distance_based", methods),
    ])
    # Same filtering and metric as plot_number_of_subgroups: all seeded subspaces identified
    df = df[df["casestudy"].isin(df[df["n_groups"] == 3]["casestudy"].unique())]
    df["identified"] = df[["f1_0", "f1_1", "f1_2"]].sum(axis=1) / df["n_groups"] == 1.0
    return identified_rate(df, "n_groups")


def load_identified(experiment):
    df = pd.concat([
        load_experiment_results_multi_method(f"{experiment}/workload", methods),
        load_experiment_results_multi_method(f"{experiment}/distance_based", methods),
    ])
    df["identified"] = df["f1"] == 1.0
    return df


@cli.command()
def stats_number_of_subgroups():
    """Statistical tests for varying the number of seeded subspaces."""
    long = load_number_of_subgroups_identified()
    omnibus, pairwise = compare_methods(long, method_names, "n_groups")
    write_stats_table(omnibus, output_dir, "rq1_number_of_subgroups_friedman", "RQ1 #subgroups: methods (Friedman)")
    write_stats_table(pairwise, output_dir, "rq1_number_of_subgroups_pairwise", "RQ1 #subgroups: methods (pairwise Wilcoxon, Holm)")
    # Pre-specified alternative: identification decreases with more seeded subspaces
    trend = trend_per_method(long, method_names, "n_groups", [1, 2, 3], "decreasing")
    write_stats_table(trend, output_dir, "rq1_number_of_subgroups_trend", "RQ1 #subgroups: trend per method (Kendall tau + Wilcoxon, Holm)")


@cli.command()
def stats_number_of_predicates():
    """Statistical tests for varying the number of predicates."""
    long = identified_rate(load_identified("number_of_predicates"), "n_conditions")
    omnibus, pairwise = compare_methods(long, method_names, "n_conditions")
    write_stats_table(omnibus, output_dir, "rq1_number_of_predicates_friedman", "RQ1 #predicates: methods (Friedman)")
    write_stats_table(pairwise, output_dir, "rq1_number_of_predicates_pairwise", "RQ1 #predicates: methods (pairwise Wilcoxon, Holm)")
    # Pre-specified alternative: identification decreases with more predicates
    levels = sorted(long["n_conditions"].unique())
    trend = trend_per_method(long, method_names, "n_conditions", levels, "decreasing")
    write_stats_table(trend, output_dir, "rq1_number_of_predicates_trend", "RQ1 #predicates: trend per method (Kendall tau + Wilcoxon, Holm)")


@cli.command()
def stats_distributions():
    """Statistical tests for varying the performance distribution of the seeded subspace."""
    long = identified_rate(load_identified("distributions"), "sg_distribution")
    omnibus, pairwise = compare_methods(long, method_names, "sg_distribution")
    write_stats_table(omnibus, output_dir, "rq1_distributions_friedman", "RQ1 distributions: methods (Friedman)")
    write_stats_table(pairwise, output_dir, "rq1_distributions_pairwise", "RQ1 distributions: methods (pairwise Wilcoxon, Holm)")
    distributions = ["normal", "bi_modal_split", "tri_modal_split", "uniform", "power_law"]
    effect = friedman_per_method(long, method_names, "sg_distribution", distributions)
    write_stats_table(effect, output_dir, "rq1_distributions_effect", "RQ1 distributions: effect per method (Friedman, Holm)")


@cli.command()
@click.pass_context
def stats_all(ctx):
    """Run all statistical tests for RQ1."""
    ctx.invoke(stats_number_of_subgroups)
    ctx.invoke(stats_number_of_predicates)
    ctx.invoke(stats_distributions)

if __name__ == "__main__":
    cli()