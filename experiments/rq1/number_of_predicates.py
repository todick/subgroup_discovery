import csv
import time
import numpy as np
import os
import sys
sys.path.insert(0, "./") 

from argparse import ArgumentParser
from sklearn.metrics import f1_score, precision_score, recall_score
from src.eval.data_gen import seed_subgroup
from src.config import Config_RQ1
from src.loaders.datasets import load
from src.methods import run_method

class ExperimentConfig:
    def __init__(self):
        self.n_rules = 5
        self.min_rule_size = 0.05
        self.max_rule_size = 0.3
        self.rel_group_width = 0.1
        self.target_dist = "normal"
        self.min_conditions = 1
        self.max_conditions = 7

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('dataset', type=str)
    parser.add_argument('casestudy', type=str)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--method', type=str, default="syflow")
    args = parser.parse_args()

    config = Config_RQ1()
    config.sd_depth = 8
    experiment_config = ExperimentConfig()
    
    data = load(args.dataset, args.casestudy)
    print(f"Running experiment with {args.method} on {args.dataset}/{args.casestudy} with seed {args.seed}")
    os.makedirs(f"results/RQ1/{args.method}/number_of_predicates/{args.dataset}/{args.casestudy}", exist_ok=True)
    result_path = f"results/RQ1/{args.method}/number_of_predicates/{args.dataset}/{args.casestudy}/{args.seed}.csv"
    with open(result_path, mode='w') as f:
        writer = csv.writer(f)
        writer.writerow(["n_conditions", "sg_mean", "sg_cardinality", "sg_rule", "best_rule", "best_rule_idx", "f1", "precision", "recall", "seed", "runtime"])
    
    for n_conditions in range(experiment_config.min_conditions, experiment_config.max_conditions+1):
        np.random.seed(args.seed)
        np.random.shuffle(data["target"])
        Y, seeded_groups, seeded_rules = seed_subgroup(
            data["data"],
            data["target"], 
            n_conditions=n_conditions, 
            min_rule_size=experiment_config.min_rule_size,
            max_rule_size=experiment_config.max_rule_size,
            target_dist=experiment_config.target_dist, 
            n_groups=1,
            rel_group_width=experiment_config.rel_group_width,
            feature_names=data["feature_names"],
            seed=args.seed)
        if not seeded_groups:
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
            data["feature_names"]
        )

        time_elapsed = time.time() - time_start
        if len(subgroups) == 0:
            f1_scores = [0] * len(seeded_groups)
            precisions = [0] * len(seeded_groups)
            recalls = [0] * len(seeded_groups)
        else:
            f1_scores = [f1_score(seeded_groups[0], sg) for sg in subgroups]
            precisions = [precision_score(seeded_groups[0], sg) for sg in subgroups]
            recalls = [recall_score(seeded_groups[0], sg) for sg in subgroups]

        best_rule = np.argmax(f1_scores)
        
        with open(result_path, mode='a') as f:
            writer = csv.writer(f)
            writer.writerow([n_conditions, sg_mean, sg_cardinality, seeded_rules[0], rules[best_rule], best_rule, f1_scores[best_rule], precisions[best_rule], recalls[best_rule], args.seed, time_elapsed])