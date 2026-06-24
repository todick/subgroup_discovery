import click
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sys
sys.path.append("./")

import src.loaders.datasets as loaders
from src.utils import reconstruct_subgroups_from_csv

from plot_config import (
    LABELFONTSIZE,
    LEGENDFONTSIZE,
    XLABELPAD,
    YLABELPAD,
    init_plot_style,
)

TICKFONTSIZE = 10
init_plot_style(style="whitegrid", font_weight="normal", axes_linewidth=0.8)

@click.group()
def cli():
    pass

@cli.command()
def plot_intro_example():
    '''Plot the example on the first page of the paper.'''
    data = loaders.load("workload", "batik")

    # Read subgroups from saved results
    results = pd.read_csv(f"results/RQ3/syflow/workload/subgroups/batik.csv")
    subgroups, rules = reconstruct_subgroups_from_csv(data, results)

    plt.figure(figsize=(6, 1.45))
    plt.xlabel("Execution time (s)", fontsize=LABELFONTSIZE, labelpad=XLABELPAD-1)
    plt.ylabel("Density", fontsize=LABELFONTSIZE, labelpad=YLABELPAD-1)
    
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_color("black")
    plt.gca().spines['bottom'].set_color("black")
    plt.grid(visible=False)
    
    ax = plt.gca()
    ax.margins(x=0.025)
    
    # Paul Tol's colorblind-friendly palette
    color_background = "#EE7733"  # Orange
    color_subspace1 = "#4477AA"   # Blue
    color_subspace2 = "#228833"   # Green
    
    # Adaptive bandwidth calculation based on data spread/width
    # Narrower distributions get higher bandwidth (inverse relationship)
    def adaptive_bw(data_subset):
        data_range = data_subset.max() - data_subset.min()
        full_range = data["target"].max() - data["target"].min()
        
        # Inverse proportionality: narrower width → higher bandwidth
        relative_width = data_range / full_range
        bw = 0.2 / relative_width  # Fixed factor of 0.15
        return max(0.1, min(bw, 2.0))  # Clamp between 0.1 and 2.0
    
    # Calculate adaptive bandwidths
    bw_full = adaptive_bw(data["target"])
    bw_sub1 = adaptive_bw(data["target"][subgroups[3]])
    bw_sub2 = adaptive_bw(data["target"][subgroups[1]])
    
    # Plot the full distribution as KDE in gray
    sns.kdeplot(data["target"], color=color_background, linewidth=0.1, label="All configurations", fill=True, alpha=0.6, bw_adjust=bw_full)
    sns.kdeplot(data["target"][subgroups[3]], color=color_subspace1, linewidth=0.1, label="Subspace 1", fill=True, alpha=0.45, bw_adjust=bw_sub1)
    sns.kdeplot(data["target"][subgroups[1]], color=color_subspace2, linewidth=0.1, label="Subspace 2", fill=True, alpha=0.45, bw_adjust=bw_sub2)
    
    plt.tick_params(axis="x", labelsize=TICKFONTSIZE, bottom=True, width=0.7, length=3)
    plt.tick_params(axis="y", labelsize=TICKFONTSIZE, left=True, width=0.7, length=3)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    
    # Legend centered above the plot
    plt.legend(
        fontsize=LEGENDFONTSIZE,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.2),
        ncol=3,
        edgecolor="white",
        frameon=False,
        handletextpad=0.6,
        handlelength=1.0,
        borderaxespad=0.2,
    )
    plt.savefig("figures/out/intro_example.pdf", bbox_inches="tight")

if __name__ == "__main__":
    cli()