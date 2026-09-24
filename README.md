# Finding Exceptional Software Configuration Subspaces with Subgroup Discovery

This repository contains supplementary material for the paper "Finding Exceptional Software Configuration Subspaces with Subgroup Discovery" submitted to ICSE 2027. 

## Running Subgroup Discovery
All datasets, subgroup discovery methods and experiment scripts needed to replicate our evaluation are provided in this repository. The required Python packages are listed in the ```requirements.txt``` file.

The ```run.ipynb``` Jupyter Notebook can be used to try various subgroup discovery methods on the data used in the paper.

## Adjustments Following the Rebuttal
We added statistical tests for all experiments for RQ1 and RQ2. The corresponding scripts and results can be found in the ```figures``` and ```figures/out/stats``` folders. All results included prior to the rebuttal remain unchanged.

## Folder Structure

- *data*
  Performance measurements collected by Mühlbauer et al. and Kaltenecker et al.
- *experiments*
  An individual script for each experiment conducted in the paper
- *figures*
  Plots and tables used in the paper as well as the code to generate them
- *results*
  Results from our evaluation. In addition to the results presented in the paper, we include results on all systems (```figures/out/full```) and real-world-subspaces found by each method (```figures/out/rq3_real_world```)
- *scripts*
  Helper scripts to run our evaluation on a SLURM cluster
- *src*
  The implementations for all subgroup discovery methods used in the paper as well as the code required for our evaluation

## Additional Results for RQ3

We provide results for all methods compared in the paper on real-world data. As RSD optimizes for finding complementary sets of subgroups, we allow up to 15 subgroups for that method. For all other methods, we only provide the first 5 subgroups found.

## Hyperparameter Sensitivity

To evaluate hyperparameter sensitivity, we conducted a one-at-a-time analysis when seeding 1-3 exceptional subspaces on the `jump3r` subject system while varying one hyperparameter at a time.

![image](figures/out/hyperparameter_sensitivity_n1.png)

The figure shows the average F1 score when seeding a single exceptional subspace; results for two and three subspaces can be found in ```figure/out/hyperparameter_sensitivity_*```. Across all tested ranges, F1 scores remain stable, confirming that the hyperparameter values chosen for RQ1 and RQ2 do not sit at a boundary or inflection point for either subgroup discovery method.

## Statistical Tests

We test all RQ1 and RQ2 results with a fixed set of tests, implemented in `figures/stats_utils.py`. Run them with
```
python figures/rq1.py stats-all
python figures/rq2.py stats-all
```
Each test family is written to `figures/out/stats/<family>.{csv,tex}`. Every test is reported, whether or not it is significant. Effect sizes are reported as numbers, without small/medium/large labels.

- **Metric:** percentage of seeds for which the seeded subspace was identified (F1 = 1), the same metric the figures use. A run that is missing for one method (timeout or crash) counts as not identified. Levels that do not exist for a system (e.g., more predicates than a system supports) are excluded.
- **Unit of analysis:** subject systems. For the RQ2 sampling method comparison, which covers only 7 systems, each system × sampling strategy combination is a unit (N = 56); units from the same system are correlated, so these p-values are somewhat optimistic. For the scalability experiments, which use a single system, the unit is seeds.
- **Method comparison** (RQ1 experiments and RQ2 sampling): Friedman omnibus test (effect size Kendall's W), then two-sided Wilcoxon signed-rank tests for all method pairs (effect size: matched-pairs rank-biserial correlation, from −1 to 1, positive when the first method scores higher).
- **Trends** (#subspaces, #predicates, t-wise coverage, random sample size): for each system, Kendall's τ between the level and the identification rate; then a one-sided Wilcoxon signed-rank test on the τ values across systems. Directions are fixed in advance: identification decreases with #subspaces and #predicates and increases with t and sample size.
- **Distributions:** Friedman test per method across the five subspace distributions.
- **Scalability:** Kendall's τ-b between #options (or #configurations) and identification or runtime, per method. Timeouts count as runtime 14400 s.
- **Multiple comparisons:** Holm–Bonferroni correction within each family (one output file), α = 0.05. We report raw and adjusted p-values.

## CART

As noted in the paper, we provide additional results for CART. By default, CART does not provide an inherent ranking of candidate subgroups. Therefore, we treat each tree node as a candidate subgroup and rank them using Kullback–Leibler divergence.
We ran our experiments for RQ1 and RQ2 (excluding scalability) using CART, visualized in ```figures/out/full```, and used CART on real world data, visualized in ```figures/out/rq3_real_world/cart_*```. 

CART achieves high F1 scores in RQ1, often outperforming Syflow and RSD. However, this performance is largely an artifact of our experimental design. The synthetic data seeds subspaces into otherwise randomized data, inducing a mean shift tied to exactly the configuration options involved in the seeded subspace. Furthermore, the seeded subspaces are independent and do not overlap, allowing CART to split on exactly the options that describe a certain subspace without being locked out of another. Consequently, our experimental setup presents a near best-case scenario for CART and does not expose its limitations as a subgroup discovery method.

These limitations become evident in the real-world setting. A qualitative analysis shows that CART predominantly identifies redundant subspaces driven by early tree splits, resulting in limited diversity, high overlap, and low aggregate coverage. In addition, CART systematically favors unimodal subspaces concentrated at distributional extremes, thereby missing broader patterns.

Taken together, these findings indicate that CART's strong performance on synthetic benchmarks does not generalize to realistic data, leading us to exclude CART from the our evaluation in the paper.