import numpy as np
import re
import pandas as pd

def get_statistics(data, subgroups, rules):
    X = data["data"]
    Y = data["target"]
    feature_names = data["feature_names"]
    
    # Statistics for individual subgroups
    subgroup_stats = []
    for i, subgroup in enumerate(subgroups):
        subgroup_stat = {}
        subgroup_stat["rule"] = rules[i]
        subgroup_stat["sg_samples"] = int(np.sum(subgroup))
        subgroup_stat["sg_coverage"] = float(np.sum(subgroup) / len(Y))
        features_used = get_features_used(rules[i], feature_names)
        subgroup_stat["rule_num_predicates"] = len(rules[i].split(" AND "))
        subgroup_stat["rule_features_used"] = list(features_used)
        subgroup_stats.append(subgroup_stat)
    subgroup_stats_df = pd.DataFrame(subgroup_stats)

    # Statistics across all subgroups
    subgroups_unique = get_unique_subgroups(subgroups)
    stats_overall = {}
    stats_overall["num_features"] = X.shape[1]
    stats_overall["num_samples"] = X.shape[0]
    ## Redundancy
    stats_overall["sgs_unique"] = len(subgroups_unique)
    stats_overall["sgs_mean_jaccard"] = float(np.mean([jaccard_index(subgroups_unique[i], subgroups_unique[j]) for i in range(len(subgroups_unique)) for j in range(i+1, len(subgroups_unique))]))
    stats_overall["sgs_mean_overlap_coefficient"] = float(np.mean([overlap_coefficient(subgroups_unique[i], subgroups_unique[j]) for i in range(len(subgroups_unique)) for j in range(i+1, len(subgroups_unique))]))
    ## Coverage
    stats_overall["coverage_mean"] = float(np.mean([np.sum(subgroup) for subgroup in subgroups_unique]) / len(Y))
    stats_overall["coverage_total"] = float(np.sum(np.any(subgroups_unique, axis=0)) / len(Y))
    ## Rule Characteristics
    features_used = set()
    for rule in rules:
        features_used.update(get_features_used(rule, feature_names))
    stats_overall["rule_features_used"] = list(features_used)
    stats_overall["rule_feature_frequency"] = [(feature, int(np.sum([1 for rule in rules if feature in get_features_used(rule, feature_names)]))) for feature in features_used]
    stats_overall["rule_num_features"] = len(features_used)
    stats_overall["rule_length_mean"] = float(np.mean([len(rule.split(" AND ")) for rule in rules]))
    stats_overall["rule_length_max"] = float(np.max([len(rule.split(" AND ")) for rule in rules]))
    stats_overall["rule_length_min"] = float(np.min([len(rule.split(" AND ")) for rule in rules]))
    overall_stats_df = pd.DataFrame([stats_overall])
    return subgroup_stats_df, overall_stats_df

def is_complementary(subgroup1, subgroup2):
    # check if two subgroups are complementary
    return np.all(subgroup1 == 1 - subgroup2) or np.all(subgroup2 == 1 - subgroup1)

def jaccard_index(subgroup1, subgroup2):
    # calculate the Jaccard index between two subgroups
    intersection = np.sum(np.logical_and(subgroup1, subgroup2))
    union = np.sum(np.logical_or(subgroup1, subgroup2))
    return intersection / union if union != 0 else 0

def overlap_coefficient(subgroup1, subgroup2):
    # calculate the overlap coefficient between two subgroups
    intersection = np.sum(np.logical_and(subgroup1, subgroup2))
    smaller_subgroup_size = min(np.sum(subgroup1), np.sum(subgroup2))
    return intersection / smaller_subgroup_size if smaller_subgroup_size != 0 else 0

def get_features_used(rule, feature_names):
    # get the features used in the rule
    features_used = set()
    for condition in rule.split(" AND "):
        condition = condition.removeprefix("NOT").strip()
        found = re.findall(r"([a-zA-Z_][a-zA-Z0-9_-]*)", condition)
        if found:
            feature_name = found[0]
            if feature_name in feature_names:
                features_used.add(feature_name)
    return features_used

def get_unique_subgroups(subgroups):
    # get the unique subgroups from the list of subgroups
    unique_subgroups = []
    for subgroup in subgroups:
        if not any(np.array_equal(subgroup, us) for us in unique_subgroups):
            unique_subgroups.append(subgroup)
    return unique_subgroups