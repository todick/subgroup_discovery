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
from src.sampling import sample_random, sample_twise

class ExperimentConfig:
    def __init__(self):
        self.n_rules = 5
        self.n_conditions = 3
        self.target_dist = "normal"
        self.min_rule_size = 0.1
        self.max_rule_size = 0.2
        self.random_sample_sizes = [2, 5, 10, 20]
        self.twise_t_values = [2, 3, 4]

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('dataset', type=str)
    parser.add_argument('casestudy', type=str)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--method', type=str, default="syflow")
    parser.add_argument('--sampling_idx', type=int, default=0, help="Index into sampling_configs list (0-7)")
    args = parser.parse_args()

    config = Config_RQ1()
    config.sd_depth = 4
    experiment_config = ExperimentConfig()
    
    # Build sampling configurations (include full to compare)
    sampling_configs = (
        [("full", None)] +
        [(f"random_{n}", n) for n in experiment_config.random_sample_sizes] +
        [(f"{t}-wise", t) for t in experiment_config.twise_t_values]
    )
    
    # Validate index
    if args.sampling_idx < 0 or args.sampling_idx >= len(sampling_configs):
        raise ValueError(f"sampling_idx must be between 0 and {len(sampling_configs) - 1}")
    
    strategy, param = sampling_configs[args.sampling_idx]
    
    data = load(args.dataset, args.casestudy)
    print(f"Running sampling strategy experiment with {args.method} on {args.dataset}/{args.casestudy} with seed {args.seed}")
    print(f"Sampling strategy: {strategy}")
    
    os.makedirs(f"results/RQ2/{args.method}/sampling/{args.dataset}/{args.casestudy}", exist_ok=True)
    result_path = f"results/RQ2/{args.method}/sampling/{args.dataset}/{args.casestudy}/{args.seed}_{args.sampling_idx}.csv"
    
    with open(result_path, mode='w') as f:
        writer = csv.writer(f)
        writer.writerow(["sampling_strategy", "sample_size", "sg_cardinality", "sg_rule", "best_rule", "best_rule_idx", "f1", "precision", "recall", "seed"])
    
    # Seed subgroups on full dataset once
    full_size = len(data["data"])
    np.random.shuffle(data["target"])
    Y_full, seeded_groups_full, seeded_rules_full = seed_subgroup(
        data["data"],
        data["target"], 
        n_conditions=experiment_config.n_conditions, 
        min_rule_size=experiment_config.min_rule_size,
        max_rule_size=experiment_config.max_rule_size,
        target_dist=experiment_config.target_dist, 
        n_groups=1,
        rel_group_width=0.2,
        feature_names=data["feature_names"],
        seed=args.seed)
    
    # Process single sampling strategy
    sample_indices = []
    while (seeded_groups_full[0][sample_indices]).sum() == 0:
        if strategy == "full":
            sample_indices = np.arange(full_size)
        elif strategy.startswith("random"):
            # Choose sample size as a multiple of the number of features
            sampled_data = sample_random(data, n_samples=data["data"].shape[1] * param, seed=args.seed)
            sample_indices = sampled_data["indices"]
        else:  # t-wise
            sampled_data = sample_twise(data, t=param, seed=args.seed)
            sample_indices = sampled_data["indices"]

    sample_size = len(sample_indices)
    X_sample = data["data"][sample_indices]
    Y_sample = Y_full[sample_indices]
    seeded_group_sample = seeded_groups_full[0][sample_indices]

    sample_subgroups, sample_rules = run_method(
        args.method,
        X_sample,
        Y_sample.reshape(-1,1), 
        config, 
        experiment_config.n_rules, 
        data["feature_names"],
    )

    print(f"Running {strategy} (sample_size={sample_size})")

    if len(sample_subgroups) > 0:
        sg_cardinality = np.sum(seeded_group_sample)

        f1_scores = [f1_score(seeded_group_sample, sg, zero_division=0) for sg in sample_subgroups]
        precisions = [precision_score(seeded_group_sample, sg, zero_division=0) for sg in sample_subgroups]
        recalls = [recall_score(seeded_group_sample, sg, zero_division=0) for sg in sample_subgroups]
        best_rule = np.argmax(f1_scores)

        with open(result_path, mode='a') as f:
            writer = csv.writer(f)
            writer.writerow([strategy, sample_size, sg_cardinality, seeded_rules_full[0], sample_rules[best_rule], best_rule, f1_scores[best_rule], precisions[best_rule], recalls[best_rule], args.seed])
    else:
        with open(result_path, mode='a') as f:
            writer = csv.writer(f)
            writer.writerow([strategy, sample_size, 0, seeded_rules_full[0], None, None, 0, 0, 0, args.seed])
