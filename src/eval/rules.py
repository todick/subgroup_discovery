import itertools
import numpy as np
from collections import defaultdict

import pandas as pd

def find_rules(X, subgroup, find_all=False):
    total_valid_rules = []
    X_in = X[subgroup]
    X_out = X[~subgroup]
    candidates = []
    for i in range(X.shape[1]):
        in_values = set(np.unique(X_in[:,i]))
        out_values = set(np.unique(X_out[:,i]))
        if in_values != out_values:
            candidates.append((i, in_values))

    predicates = []
    for f, values in candidates:
        predicates.extend([(f, v) for v in values])
    
    for i in range(1, len(predicates)+1):
        valid_rules = []
        for subset in itertools.combinations(predicates, i):
            rule = [p for p in subset]
            if _rule_is_enough(X_in, X_out, rule):
                valid_rules.append(subset)
        if valid_rules:
            total_valid_rules.extend(valid_rules)
            if not find_all:
                break
    return total_valid_rules

def find_observable_rules(X, subgroup, find_all=False):
    candidates = find_rules(X, subgroup, True)
    X_in = X[subgroup]
    X_out = X[~subgroup]
    observable_rules = []
    
    for rule in candidates:
        if _check_observability(X_in, X_out, rule):
            if not find_all:
                return [rule]
            observable_rules.append(rule)
    return observable_rules
            
def _rule_is_enough(X_in, X_out, rule):
    # check if the rule is enough to explain the difference between X_in and X_out
    # rules can also contain multiple assignments for the same feature
    rule_dict = defaultdict(set)
    for f, v in rule:
        rule_dict[f].add(v)

    # check if all configurations in X_in satisfy the rule
    for f, values in rule_dict.items():
        if not np.all(np.isin(X_in[:,f], list(values))):
            return False

    # check if all configurations in X_out do not satisfy the rule
    for i in range(X_out.shape[0]):
        if all([X_out[i][f] in values for f, values in rule_dict.items()]):
            return False
    return True

def _check_observability(X_in, X_out, rule):
    # if there are multiple values for the same feature, we need to split the rule
    # into multiple rules and check each of them separately
    rule_dict = defaultdict(set)
    for f, v in rule:
        rule_dict[f].add(v)
        
    rules = [[]]
    for f, values in rule_dict.items():
        rules = [r + [(f, v)] for r in rules for v in values]

    for rule in rules:
        restricted_features = {}
        for f, v in rule:
            restricted_features[f] = rule_dict[f] - set([v])

        partitions = _get_partitions(rule)
        if not any(_partition_is_valid(X_in, X_out, partition, restricted_features) for partition in partitions):
            return False
                
    return True 

def _partition_is_valid(X_in, X_out, partition, restricted_features):
    for partial_rule in partition:
        if not _find_witnesses(X_in, X_out, partial_rule, restricted_features):
            return False
    return True

def _get_partitions(rule):
    if not rule:
        return [[]]
    first = rule[0]
    rest_partitions = _get_partitions(rule[1:])
    result = []
    for partition in rest_partitions:
        for i in range(len(partition)):
            new_partition = partition[:i] + [partition[i] + [first]] + partition[i+1:]
            result.append(new_partition)
        result.append([[first]] + partition)
    return result

def _find_witnesses(X_in, X_out, rule, restricted_features):       
    unaffected_features = set(range(X_in.shape[1])) - set([f for f, _ in rule])
    rule_features = np.array([f for f, _ in rule])
    rule_values = np.array([v for _, v in rule])

    witnesses = []
    for i in range(X_in.shape[0]):
        witness = X_in[i]
        if np.all(witness[rule_features] == rule_values):
            witnesses.append(witness)

    for i in range(X_out.shape[0]):
        counterwitness = X_out[i]
        if np.any(counterwitness[rule_features] == rule_values):
            continue
        if not all([counterwitness[f] == witness[f] for f in unaffected_features]):
            continue
        if any(counterwitness[f] in restricted_features[f] for f in rule_features):
            continue
        return True
    return False

def pretty_rule(rule, feature_names):
    # if multiple values for the same feature are present, group them
    rule_dict = defaultdict(set)
    for f, v in rule:
        rule_dict[f].add(v)    
    pretty_rule = [f"{feature_names[f]} in {v}" for f, v in rule_dict.items()]

    pretty_rule = " AND ".join(pretty_rule)
    return pretty_rule