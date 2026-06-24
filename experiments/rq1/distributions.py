import csv
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
        self.n_conditions = 3
        self.min_rule_size = 0.1
        self.max_rule_size = 0.2
        self.target_properties = [{
            "dist": "normal",
            "rel_group_width": 0.2
        },
        {
            "dist": "uniform",
            "rel_group_width": 0.5
        },
        {
            "dist": "bi_modal_split",
            "rel_group_width": 0.05
        },
        {
            "dist": "tri_modal_split",
            "rel_group_width": 0.025
        },
        {
            "dist": "power_law",
            "rel_group_width": 1
        }]

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('dataset', type=str)
    parser.add_argument('casestudy', type=str)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--method', type=str, default="syflow")
    parser.add_argument('--target_idx', type=int, default=0, help="Index into target_properties list (0-4)")
    args = parser.parse_args()

    config = Config_RQ1()
    config.sd_depth = 4
    experiment_config = ExperimentConfig()
    
    # Validate index
    if args.target_idx < 0 or args.target_idx >= len(experiment_config.target_properties):
        raise ValueError(f"target_idx must be between 0 and {len(experiment_config.target_properties) - 1}")
    
    target_props = experiment_config.target_properties[args.target_idx]
    
    data = load(args.dataset, args.casestudy)
    print(f"Running distribution experiment with {args.method} on {args.dataset}/{args.casestudy} with seed {args.seed}")
    print(f"Target properties: {target_props}")
    
    os.makedirs(f"results/RQ1/{args.method}/distributions/{args.dataset}/{args.casestudy}", exist_ok=True)
    result_path = f"results/RQ1/{args.method}/distributions/{args.dataset}/{args.casestudy}/{args.seed}_{args.target_idx}.csv"
    
    with open(result_path, mode='w') as f:
        writer = csv.writer(f)
        writer.writerow(["sg_distribution", "sg_cardinality", "sg_rule", "best_rule", "best_rule_idx", "f1", "precision", "recall", "seed"])
    
    # Use full dataset (no sampling)
    full_size = len(data["data"])
    np.random.shuffle(data["target"])
    Y_full, seeded_groups_full, seeded_rules_full = seed_subgroup(
        data["data"],
        data["target"], 
        n_conditions=experiment_config.n_conditions, 
        min_rule_size=experiment_config.min_rule_size,
        max_rule_size=experiment_config.max_rule_size,
        target_dist=target_props["dist"], 
        n_groups=1,
        rel_group_width=target_props["rel_group_width"],
        feature_names=data["feature_names"],
        seed=args.seed)
    
    X_full = data["data"]
    seeded_group_full = seeded_groups_full[0]

    sample_subgroups, sample_rules = run_method(
        args.method,
        X_full,
        Y_full.reshape(-1,1), 
        config, 
        experiment_config.n_rules, 
        data["feature_names"],
    )

    print(f"Running on full dataset (size={full_size})")

    if len(sample_subgroups) > 0:
        sg_cardinality = np.sum(seeded_group_full)

        f1_scores = [f1_score(seeded_group_full, sg, zero_division=0) for sg in sample_subgroups]
        precisions = [precision_score(seeded_group_full, sg, zero_division=0) for sg in sample_subgroups]
        recalls = [recall_score(seeded_group_full, sg, zero_division=0) for sg in sample_subgroups]
        best_rule = np.argmax(f1_scores)

        with open(result_path, mode='a') as f:
            writer = csv.writer(f)
            writer.writerow([target_props["dist"], sg_cardinality, seeded_rules_full[0], sample_rules[best_rule], best_rule, f1_scores[best_rule], precisions[best_rule], recalls[best_rule], args.seed])
    else:
        with open(result_path, mode='a') as f:
            writer = csv.writer(f)
            writer.writerow([target_props["dist"], 0, seeded_rules_full[0], None, None, 0, 0, 0, args.seed])
