import os
import sys
import torch
sys.path.insert(0, "./") 

from argparse import ArgumentParser
from src.config import Config_RQ3
from src.eval.statistics import get_statistics
from src.loaders.datasets import load
from src.methods import run_method

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('dataset', type=str)
    parser.add_argument('casestudy', type=str)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--method', type=str, default="rsd")
    args = parser.parse_args()

    config = Config_RQ3()

    data = load(args.dataset, args.casestudy)
    print(f"Finding subgroups in {args.dataset}/{args.casestudy}")
    
    subgroups, rules = run_method(
        args.method,
        data["data"], 
        data["target"].reshape(-1,1), 
        config, 
        10,
        data["feature_names"]
    )

    subgroup_stats, overall_stats = get_statistics(data, subgroups, rules)
    path = f"results/RQ3/{args.method}/{args.dataset}/"
    os.makedirs(f"{path}/stats", exist_ok=True)
    os.makedirs(f"{path}/subgroups", exist_ok=True)

    overall_stats.to_csv(f"{path}/stats/{args.casestudy}.csv", index=False)
    subgroup_stats.to_csv(f"{path}/subgroups/{args.casestudy}.csv", index=False)
    print(f"Results saved to {path}/stats/{args.casestudy}.csv and {path}/subgroups/{args.casestudy}.csv")
    