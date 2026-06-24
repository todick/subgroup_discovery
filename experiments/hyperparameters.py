import csv
import time
import numpy as np
import os
import sys
sys.path.insert(0, "./") 

from argparse import ArgumentParser
from sklearn.metrics import f1_score, precision_score, recall_score
from src.config import Config_RQ1
from src.eval.data_gen import seed_subgroup
from src.loaders.datasets import load
from src.methods import run_method

class ExperimentConfig:
    def __init__(self):
        self.n_rules = 5
        self.n_conditions = 2
        self.min_rule_size = 0.1
        self.max_rule_size = 0.2
        self.rel_group_width = 0.1
        self.target_dist = "normal"
        self.max_overlap = 0.05
        self.groups_min = 1
        self.groups_max = 3

class HyperparameterGrid:
    def __init__(self):
        self.grid = {
            "alpha": [0.1, 0.3, 0.5, 0.75, 1.0],
            "lambd": [0, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
            "beam_width": [20, 40, 80, 160],
            "rsd_min_support": [0.01, 0.025, 0.05, 0.1]
        }

    def get_parameters_for_method(self, method):
        if method == "syflow":
            return {k: v for k, v in self.grid.items() if k in ["alpha", "lambd"]}
        elif method == "beam-kl":
            return {k: v for k, v in self.grid.items() if k in ["alpha", "beam_width"]}
        elif method == "beam-mean":
            return {k: v for k, v in self.grid.items() if k in ["alpha", "beam_width"]}
        elif method == "dfs-mean":
            return {k: v for k, v in self.grid.items() if k in ["alpha"]}
        elif method == "rsd":
            return {k: v for k, v in self.grid.items() if k in ["beam_width", "rsd_min_support"]}
        raise ValueError(f"Unknown method: {method}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('dataset', type=str)
    parser.add_argument('casestudy', type=str)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--method', type=str, default="syflow")
    args = parser.parse_args()
    experiment_config = ExperimentConfig()
    
    data = load(args.dataset, args.casestudy)
    print(f"Running experiment with {args.method} on {args.dataset}/{args.casestudy} with seed {args.seed}")
    os.makedirs(f"results/hyperparameter_sensitivity/{args.method}/{args.dataset}/{args.casestudy}", exist_ok=True)
    result_path = f"results/hyperparameter_sensitivity/{args.method}/{args.dataset}/{args.casestudy}/{args.seed}.csv"

    grid = HyperparameterGrid().get_parameters_for_method(args.method)

    with open(result_path, mode='w') as f:
        writer = csv.writer(f)
        column_names = ["param_name", "param_value", "n_groups"]
        for i in range(experiment_config.groups_max):
            column_names += [f"f1_{i}", f"precision_{i}", f"recall_{i}"]
        column_names += ["seed", "runtime"]
        writer.writerow(column_names)

    for param_name, param_values in grid.items():
        for param_value in param_values:
            config = Config_RQ1()
            setattr(config, param_name, param_value)
            print(f"Running experiment with {args.method} on {args.dataset}/{args.casestudy} with seed {args.seed} and {param_name}={param_value}")

            for n_groups in range(experiment_config.groups_min, experiment_config.groups_max+1):
                np.random.seed(args.seed)
                np.random.shuffle(data["target"])
                print(f"Running experiment with {n_groups} subgroups.")
                for i in range(1,50):
                    Y, seeded_groups, seeded_rules = seed_subgroup(
                        data["data"],
                        data["target"],
                        n_conditions=experiment_config.n_conditions,
                        min_rule_size=experiment_config.min_rule_size,
                        max_rule_size=experiment_config.max_rule_size,
                        target_dist=experiment_config.target_dist,
                        n_groups=n_groups,
                        rel_group_width=experiment_config.rel_group_width,
                        max_overlap=experiment_config.max_overlap,
                        feature_names=data["feature_names"],
                        n_tries=100000,
                        seed=None)
                    if seeded_groups is None:
                        continue
                    if len(seeded_groups) == n_groups:
                        break
                if seeded_groups is None or len(seeded_groups) < n_groups:
                    print(f"Could not seed {n_groups} subgroups.")
                    continue
                sg_mean = np.mean(Y[seeded_groups[0]])
                sg_cardinality = np.sum(seeded_groups[0])

                time_start = time.time()
                subgroups, rules = run_method(
                    args.method,
                    data["data"],
                    Y.reshape(-1,1),
                    config,
                    experiment_config.n_rules,
                    data["feature_names"],
                )
                time_elapsed = time.time() - time_start

                f1_scores = []
                precisions = []
                recalls = []
                if len(subgroups) == 0:
                    best_rule_idx = 0
                    f1_scores = [0] * len(seeded_groups)
                    precisions = [0] * len(seeded_groups)
                    recalls = [0] * len(seeded_groups)
                else:
                    for i in range(n_groups):
                        best_rule_idx = np.argmax([f1_score(seeded_groups[i], sg) for sg in subgroups])
                        f1_scores.append(f1_score(seeded_groups[i], subgroups[best_rule_idx]))
                        precisions.append(precision_score(seeded_groups[i], subgroups[best_rule_idx]))
                        recalls.append(recall_score(seeded_groups[i], subgroups[best_rule_idx]))

                with open(result_path, mode='a') as f:
                    writer = csv.writer(f)
                    row = [param_name, param_value, n_groups]
                    for i in range(n_groups):
                        row += [f1_scores[i], precisions[i], recalls[i]]
                    for i in range(n_groups, experiment_config.groups_max):
                        row += [0, 0, 0]
                    row += [args.seed, time_elapsed]
                    writer.writerow(row)