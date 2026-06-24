"""Shared plotting configuration for all figures."""
import colorsys
import glob
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns

# Color and font size constants
COLOR_PALETTE = "muted"
LABELFONTSIZE = 14
LEGENDFONTSIZE = 12
TICKFONTSIZE = 12
LEGENDTITLEFONTSIZE = 16
XLABELPAD = 6
YLABELPAD = 6
MARKERLINEWIDTH = 0.7

# System name mappings for LaTeX formatting
system_names = {
    "batik": "\\textsc{batik}",
    "dconvert": "\\textsc{dconvert}",
    "h2": "\\textsc{h2}",
    "jump3r": "\\textsc{jump3r}",
    "kanzi": "\\textsc{kanzi}",
    "lrzip": "\\textsc{lrzip}",
    "x264": "\\textsc{x264}",
    "xz": "\\textsc{xz}",
    "z3": "\\textsc{z3}",
    "7z": "\\textsc{7z}",
    "BerkeleyDBC": "\\textsc{BDB-C}",
    "Dune": "\\textsc{Dune}",
    "Hipacc": "\\textsc{HIPA$^{cc}$}",
    "JavaGC": "\\textsc{JavaGC}",
    "LLVM": "\\textsc{LLVM}",
    "LRZip": "\\textsc{lrzip}",
    "Polly": "\\textsc{Polly}",
    "VP9": "\\textsc{VP9}",
}


def init_plot_style(style="whitegrid", font_weight="bold", axes_linewidth=0.5):
    """Initialize matplotlib and seaborn plotting style.
    
    Args:
        style: Seaborn style (default: "whitegrid")
        font_weight: Font weight for labels (default: "bold")
        axes_linewidth: Width of axes lines (default: 0.5)
    """
    sns.set_theme(style=style)
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "text.latex.preamble": "\\usepackage{mathptmx} \\usepackage{eucal} \\usepackage{newtxtext, newtxmath}",
        "mathtext.fontset": "cm",
        "axes.labelsize": LABELFONTSIZE,
        "axes.linewidth": axes_linewidth,
        "lines.linewidth": 1.0,
        "font.weight": font_weight,
        "axes.labelweight": font_weight,
    })

def load_experiment_results(folder):
    """Load experiment results from CSV files in a folder.
    
    Args:
        folder: Path to folder containing CSV files
        
    Returns:
        DataFrame with combined results from all CSV files
    """
    casestudies = glob.glob(f"{folder}/*.csv")
    if not casestudies:
        return pd.DataFrame()

    dfs = [pd.read_csv(casestudy) for casestudy in casestudies]
    for casestudy, df in zip(casestudies, dfs):
        df["casestudy"] = casestudy.split("/")[-1].split("\\")[-1].split(".")[0]
    dfs = [df for df in dfs if not df.empty]
    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs)
    df["casestudy"] = df["casestudy"].replace(system_names)
    if "workload" in folder:
        df = df[df["casestudy"] != "\\textsc{lrzip}"]
    if "distance_based" in folder:
        df = df[df["casestudy"] != "\\textsc{lrzip}"]
        df = df[df["casestudy"] != "\\textsc{x264}"]
    return df

def darken_color(color, factor=0.5):
    """Return a darker version of the given color by reducing lightness."""
    r, g, b = mcolors.to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = max(0, l * factor)
    return colorsys.hls_to_rgb(h, l, s)

def set_scatter_edgecolors(ax):
    """Set scatter plot edge colors to darker versions of face colors with full opacity."""
    for collection in ax.collections:
        if hasattr(collection, 'get_facecolors') and len(collection.get_facecolors()) > 0:
            facecolors = collection.get_facecolors()
            # Clear collection-level alpha so per-vertex alpha values are used
            collection.set_alpha(None)
            # Restore facecolors with their embedded alpha
            collection.set_facecolors(facecolors)
            # Edge colors are darker and at full opacity (alpha=1.0)
            edgecolors = np.array([darken_color(fc[:3]) + (1.0,) for fc in facecolors])
            collection.set_edgecolors(edgecolors)

def get_edge_colors(palette):
    """Return edge colors as darker versions of the palette colors."""
    return [darken_color(c) for c in palette]