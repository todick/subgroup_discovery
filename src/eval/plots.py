import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from matplotlib.colors import to_rgba

plt.rcParams['font.family'] = 'serif'

def plot_target(target, label_x, label_y, title, bins=200, density=True):
    plt.figure(figsize=(10, 6))
    plt.xlabel(label_x, fontsize=16, labelpad=16)
    plt.ylabel(label_y, fontsize=16, labelpad=16)
    plt.title(title, fontsize=18, fontweight='bold', pad=20)
    
    plt.rc('lines', linewidth=1)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.grid(visible=True, which='major', linestyle='--', linewidth=0.5, alpha=0.7)
    bins = np.histogram(target, bins=bins)[1]
    fill_color = to_rgba(plt.cm.tab10.colors[0], alpha=0.35)
    edge_color = to_rgba(plt.cm.tab10.colors[0], alpha=0.6)
    plt.hist(target, bins, label="Full distribution", color=fill_color, edgecolor=edge_color, density=density)
    plt.show()

def plot_subgroups(target, subgroups, rules, which_sgs, label_x, label_y, title, bins=200, legend=True, stack=True):  
    plt.figure(figsize=(10, 6))
    plt.xlabel(label_x, fontsize=16, labelpad=16)
    plt.ylabel(label_y, fontsize=16, labelpad=16)
    plt.title(title, fontsize=18, fontweight='bold', pad=20)
    
    plt.rc('lines', linewidth=1)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.grid(visible=True, which='major', linestyle='--', linewidth=0.5, alpha=0.7)
    
    bins = np.histogram(target, bins=bins)[1]
    plt.hist(target, bins, label="Full distribution", alpha=0.3, color='gray')
    
    subgroup_histograms = []
    for sg in which_sgs:
        subgroup = target[subgroups[sg]]
        hist, _ = np.histogram(subgroup, bins)
        subgroup_histograms.append(hist)
    
    bottom = np.zeros_like(subgroup_histograms[0])
    for hist, sg in zip(subgroup_histograms, which_sgs):
        fill_color = to_rgba(plt.cm.tab10.colors[sg], alpha=0.35)
        edge_color = to_rgba(plt.cm.tab10.colors[sg], alpha=0.6)
        plt.bar(
            bins[:-1], hist, width=np.diff(bins), bottom=bottom, 
            # darker edge color, but same color as fill
            label=rules[sg], align='edge', color=fill_color, edgecolor=edge_color
        )
        if stack:
            bottom += hist
    
    if legend:
        plt.legend(fontsize=12, loc='upper right', fancybox=True)
    plt.tight_layout()
    plt.show()

def plot_tsne_subgroups(features, subgroups, rules, which_sgs, title, perplexity=30, random_state=42):
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    tsne_result = tsne.fit_transform(features)

    plt.figure(figsize=(10, 6))
    plt.title(title, fontsize=18, fontweight='bold', pad=20)
    plt.xlabel("T-SNE Component 1", fontsize=16, labelpad=16)
    plt.ylabel("T-SNE Component 2", fontsize=16, labelpad=16)
    plt.grid(visible=True, which='major', linestyle='--', linewidth=0.5, alpha=0.7)

    for sg in which_sgs:
        mask = subgroups[sg]
        plt.scatter(
            tsne_result[mask, 0], tsne_result[mask, 1],
            label=rules[sg], alpha=0.7, s=50,
            color=plt.cm.tab10.colors[sg]
        )

    # plot the full distribution
    plt.scatter(
        tsne_result[np.ones(len(features), dtype=bool), 0], tsne_result[np.ones(len(features), dtype=bool), 1],
        label="Full distribution", alpha=0.3, s=10, color='gray'
    )

    # put legend above the plot
    plt.legend(fontsize=12, loc='upper center', bbox_to_anchor=(0.5, 1.25), fancybox=True, ncol=2)
    plt.tight_layout()
    plt.show()