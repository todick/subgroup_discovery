"""Statistical tests for RQ1 and RQ2.

All tests follow the same, pre-registered protocol:
- The metric is the percentage of seeds for which a seeded subspace was identified (F1 = 1).
- The unit of analysis is the subject system (percentage over its seeds); for scalability
  experiments, which use a single system, individual seeds are the unit of analysis.
- Each output table is one family of hypotheses. P-values are corrected with Holm-Bonferroni
  within each family; raw and adjusted p-values are always reported, significant or not.
- A run that is missing for one method (timeout or crash) counts as not identified; levels or
  seeds that no method ran are structural gaps and are excluded (see `identified_rate`).
"""
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ALPHA = 0.05


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (NaN p-values are treated as 1)."""
    p = np.nan_to_num(np.asarray(pvals, dtype=float), nan=1.0)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        running_max = max(running_max, (m - rank) * p[idx])
        adjusted[idx] = min(1.0, running_max)
    return adjusted


def rank_biserial(x, y):
    """Matched-pairs rank-biserial correlation for paired samples x and y.

    Computed from the signed ranks of the Wilcoxon signed-rank test (Pratt: zero differences
    are ranked but belong to neither side): r = (R+ - R-) / (n(n+1)/2). Ranges from -1 (y always
    larger) to 1 (x always larger); 0 means no consistent difference.
    """
    diff = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    n = len(diff)
    if n == 0:
        return np.nan
    ranks = stats.rankdata(np.abs(diff))
    return (ranks[diff > 0].sum() - ranks[diff < 0].sum()) / (n * (n + 1) / 2)


def identified_rate(df, level_col):
    """Percentage of identified seeds per (casestudy, method, level); expects a boolean 'identified' column.

    The seeds of a (casestudy, level) cell are all seeds that any method ran. A seed missing for
    one method (timeout or crash) counts as not identified. Levels or seeds no method ran are
    structural gaps and are excluded. Methods without any result for a system are excluded for
    that system.
    """
    df = df[["casestudy", "method", level_col, "seed", "identified"]].copy()
    df["method"] = df["method"].astype(str)
    cells = df[["casestudy", level_col, "seed"]].drop_duplicates()
    systems = df[["casestudy", "method"]].drop_duplicates()
    grid = cells.merge(systems, on="casestudy")
    grid = grid.merge(df, on=["casestudy", "method", level_col, "seed"], how="left")
    grid["identified"] = grid["identified"].fillna(False).astype(bool)
    return (
        grid.groupby(["casestudy", "method", level_col])["identified"]
        .mean()
        .mul(100)
        .rename("value")
        .reset_index()
    )


def _wilcoxon(x, y, alternative="two-sided"):
    diff = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    if np.allclose(diff, 0):
        return np.nan, 1.0
    res = stats.wilcoxon(x, y, zero_method="pratt", alternative=alternative)
    return res.statistic, res.pvalue


def _friedman(wide):
    """Friedman test over the columns of a (blocks x treatments) table, plus Kendall's W."""
    n, k = wide.shape
    if n < 2 or np.allclose(wide.to_numpy(), wide.to_numpy()[:, :1]):
        return np.nan, 1.0, 0.0
    chi2, p = stats.friedmanchisquare(*[wide[c].to_numpy() for c in wide.columns])
    if not np.isfinite(chi2):
        return np.nan, 1.0, 0.0
    return chi2, p, chi2 / (n * (k - 1))


def compare_methods(long, method_names, level_col, level_blocks=False):
    """Compare all methods on identification rates.

    By default, each system is one block and its value is averaged over all levels of
    `level_col` that system has. With `level_blocks=True`, each (system, level) cell is its own
    block; this raises N when there are few systems, at the cost of blocks from the same system
    being correlated. Blocks without a result for one of the methods are dropped. Runs a
    Friedman omnibus test and two-sided Wilcoxon signed-rank tests for all method pairs; Holm
    correction over the pairs.

    Returns (omnibus, pairwise) DataFrames.
    """
    methods = [m for m in method_names if m in set(long["method"])]
    blocks = ["casestudy", level_col] if level_blocks else ["casestudy"]
    per_system = long.groupby([*blocks, "method"])["value"].mean().unstack()[methods].dropna()

    chi2, p, w = _friedman(per_system)
    omnibus = pd.DataFrame([{
        "test": "Friedman",
        "n": len(per_system),
        "k": len(methods),
        "statistic": chi2,
        "p": p,
        "kendall_w": w,
        **{f"mean {m}": per_system[m].mean() for m in methods},
    }])

    rows = []
    for a, b in combinations(methods, 2):
        stat, p = _wilcoxon(per_system[a], per_system[b])
        rows.append({
            "method_a": a,
            "method_b": b,
            "n": len(per_system),
            "median_a": per_system[a].median(),
            "median_b": per_system[b].median(),
            "statistic": stat,
            "p": p,
            "rank_biserial": rank_biserial(per_system[a], per_system[b]),
        })
    pairwise = _finalize(pd.DataFrame(rows))
    return omnibus, pairwise


def trend_per_method(long, method_names, level_col, order, direction):
    """Monotonic trend test per method, with systems as the unit of analysis.

    For each system, Kendall's tau-b between the level (ranked by `order`) and the
    identification rate is computed over the levels that system has (systems differ in which
    levels exist). A one-sided Wilcoxon signed-rank test then checks whether the per-system taus
    are shifted in the pre-specified `direction` ("increasing" or "decreasing"). Systems with a
    constant rate contribute tau = 0. Holm correction over methods.
    """
    rank = {level: i for i, level in enumerate(order)}
    alternative = "greater" if direction == "increasing" else "less"
    rows = []
    for method in method_names:
        sub = long[(long["method"] == method) & long[level_col].isin(order)]
        if sub.empty:
            continue
        taus = []
        for _, system in sub.groupby("casestudy"):
            if system[level_col].nunique() < 2:
                continue
            tau = stats.kendalltau(system[level_col].map(rank), system["value"]).correlation
            taus.append(0.0 if np.isnan(tau) else tau)
        taus = np.asarray(taus)
        if len(taus) == 0 or np.allclose(taus, 0):
            stat, p = np.nan, 1.0
        else:
            res = stats.wilcoxon(taus, zero_method="pratt", alternative=alternative)
            stat, p = res.statistic, res.pvalue
        means = sub.groupby(level_col)["value"].mean()
        rows.append({
            "method": method,
            "n": len(taus),
            "direction": direction,
            **{f"mean {level}": means.get(level, np.nan) for level in order},
            "median_tau": np.median(taus) if len(taus) else np.nan,
            "statistic": stat,
            "p": p,
        })
    return _finalize(pd.DataFrame(rows))


def friedman_per_method(long, method_names, level_col, levels):
    """Friedman test per method across the (unordered) levels of `level_col`, systems as blocks."""
    rows = []
    for method in method_names:
        sub = long[long["method"] == method]
        if sub.empty:
            continue
        wide = sub.pivot_table(index="casestudy", columns=level_col, values="value", observed=True)
        wide = wide.reindex(columns=levels).dropna()
        chi2, p, w = _friedman(wide)
        rows.append({
            "method": method,
            "n": len(wide),
            **{f"mean {level}": wide[level].mean() for level in levels},
            "statistic": chi2,
            "p": p,
            "kendall_w": w,
        })
    return _finalize(pd.DataFrame(rows))


def kendall_trend_per_method(df, method_names, level_col, value_col):
    """Two-sided Kendall's tau-b between `level_col` and `value_col` per method (seeds as unit)."""
    rows = []
    for method in method_names:
        sub = df[df["method"] == method]
        if sub.empty:
            continue
        if sub[value_col].nunique() < 2:
            tau, p = np.nan, 1.0
        else:
            tau, p = stats.kendalltau(sub[level_col], sub[value_col])
        rows.append({
            "method": method,
            "n": len(sub),
            **{f"mean {level}": value for level, value in sub.groupby(level_col)[value_col].mean().items()},
            "tau": tau,
            "p": p,
        })
    return _finalize(pd.DataFrame(rows))


def _finalize(df):
    if df.empty:
        return df
    df["p_holm"] = holm(df["p"])
    df["significant"] = df["p_holm"] < ALPHA
    return df


def _format_p(p):
    if pd.isna(p):
        return "--"
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def write_stats_table(df, output_dir, name, title=None):
    """Write a family's results to <output_dir>/stats/<name>.{csv,tex} and print them."""
    out = Path(output_dir) / "stats"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{name}.csv", index=False)

    tex = df.copy()
    for col in tex.columns:
        if col in ("p", "p_holm"):
            tex[col] = tex[col].map(_format_p)
        elif col == "significant":
            tex[col] = tex[col].map(lambda s: "\\checkmark" if s else "")
    tex.columns = [str(c).replace("_", "\\_").replace("%", "\\%") for c in tex.columns]
    tex.to_latex(
        out / f"{name}.tex",
        index=False,
        float_format=lambda x: f"{x:.2f}",
        escape=False,
        na_rep="--",
    )

    print(f"=== {title or name} ===")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print()
