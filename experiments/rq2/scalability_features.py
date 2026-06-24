import csv
import time
import numpy as np
import os
import sys
sys.path.insert(0, "./") 

from argparse import ArgumentParser
from sklearn.metrics import f1_score, precision_score, recall_score
from src.config import Config_RQ1
from src.eval.data_gen import scale_and_seed_subgroup
from src.loaders.datasets import load
from src.methods import run_method

class ExperimentConfig:
    def __init__(self):
        self.n_rules = 5
        self.n_conditions = 4
        self.target_dist = "normal"
        self.n_features = [10, 25, 50, 100, 250, 500, 1000]
        self.n_samples = 10000

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('dataset', type=str)
    parser.add_argument('casestudy', type=str)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--method', type=str, default="syflow")
    args = parser.parse_args()

    config = Config_RQ1()
    experiment_config = ExperimentConfig()
    
    data = load(args.dataset, args.casestudy)
    print(f"Running experiment with {args.method} on {args.dataset}/{args.casestudy} with seed {args.seed}")
    os.makedirs(f"results/RQ2/{args.method}/scalability_features/{args.dataset}/{args.casestudy}", exist_ok=True)
    result_path = f"results/RQ2/{args.method}/scalability_features/{args.dataset}/{args.casestudy}/{args.seed}.csv"
    with open(result_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["n_samples", "n_features", "sg_mean", "sg_cardinality", "sg_rule", "best_rule", "best_rule_idx", "f1", "precision", "recall", "seed", "runtime"])

    for n_features in experiment_config.n_features:
        X, Y, seeded_groups, seeded_rules = scale_and_seed_subgroup(
            experiment_config.n_samples, 
            n_features, 
            data["target"].reshape(-1, 1), 
            numeric = 0.2,
            sg_constraints=experiment_config.n_conditions,
            sg_dist=experiment_config.target_dist,
        )
        feature_names = [f"X_{i}" for i in range(n_features)]
        sg_mean = np.mean(Y[seeded_groups[0]])
        sg_cardinality = np.sum(seeded_groups[0])

        time_start = time.time()
        subgroups, rules = run_method(
            args.method,
            X,
            Y.reshape(-1,1), 
            config, 
            experiment_config.n_rules, 
            feature_names, 
        )
        time_elapsed = time.time() - time_start
        
        if len(subgroups) == 0:
            f1_scores = [0] * len(seeded_groups)
            precisions = [0] * len(seeded_groups)
            recalls = [0] * len(seeded_groups)
            rules = [""] * len(seeded_groups)
        else:
            f1_scores = [f1_score(seeded_groups[0], sg) for sg in subgroups]
            precisions = [precision_score(seeded_groups[0], sg) for sg in subgroups]
            recalls = [recall_score(seeded_groups[0], sg) for sg in subgroups]
        best_rule = np.argmax(f1_scores)
        
        with open(result_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([X.shape[0], X.shape[1], sg_mean, sg_cardinality, seeded_rules[0], rules[best_rule], best_rule, f1_scores[best_rule], precisions[best_rule], recalls[best_rule], args.seed, time_elapsed])