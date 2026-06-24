import numpy as np
import pandas as pd
import re
import operator

from scipy.stats import gaussian_kde, wasserstein_distance, energy_distance
    
def wkl(Y, Y_subgroup, binary,a=1/2):
    if binary:
        y0 = np.sum(Y==0)/Y.shape[0]+1e-6
        y1 = np.sum(Y==1)/Y.shape[0]+1e-6
        ys0 = np.sum(Y_subgroup==0)/Y_subgroup.shape[0]+1e-6
        ys1 = np.sum(Y_subgroup==1)/Y_subgroup.shape[0]+1e-6
        kl = ys0*np.log(ys0/y0) + ys1*np.log(ys1/y1)
        return kl*(Y_subgroup.shape[0]/Y.shape[0])
    else:
        density_y_subgroup = gaussian_kde(Y_subgroup.T)
        density_y = gaussian_kde(Y.T)
        log_y_s = density_y_subgroup.logpdf(Y_subgroup.T)
        log_y = density_y.logpdf(Y_subgroup.T)
        p_y_s = density_y_subgroup.pdf(Y_subgroup.T)
        n_subgroup = Y_subgroup.shape[0]
        kl = np.sum(p_y_s*(log_y_s-log_y))/np.sum(p_y_s)
        if kl < 0:
            kl = 0
        return kl*(n_subgroup/Y.shape[0])**a

def evaluate_overlap(subgroup_labels):
    overlap = 0
    n = 0
    for i in range(len(subgroup_labels)):
        for j in range(i+1,len(subgroup_labels)):
            label1 = subgroup_labels[i]
            label2 = subgroup_labels[j]
            overlap += np.sum(np.logical_and(label1,label2))/np.sum(np.logical_or(label1,label2))
            n += 1
    if n == 0:
        return 0
    return overlap/n

def compute_tv(Y, Y_subgroup,a=1/2):
    density_y_subgroup = gaussian_kde(Y_subgroup.T)
    density_y = gaussian_kde(Y.T)
    p_y_s = density_y_subgroup.pdf(Y_subgroup.T)
    p_y = density_y.pdf(Y_subgroup.T)
    tv = np.sum(np.abs(p_y_s-p_y))
    return tv*(Y_subgroup.shape[0]/Y.shape[0])**a

def compute_wd(Y, Y_subgroup,a=1/2):
    density_y_subgroup = gaussian_kde(Y_subgroup.T)
    density_y = gaussian_kde(Y.T)
    rYs = density_y_subgroup.resample(size=(20000,))
    Ys = density_y.resample(size=(20000,))
    rYs = rYs.reshape((20000,))
    Ys = Ys.reshape((20000,))
    #weights_sub = (rYs < Y_subgroup.max()) + (rYs > Y_subgroup.min())
    #weights = (Ys < Y_subgroup.max()) + (Ys > Y_subgroup.min())
    mask  = np.logical_and(rYs <= Y_subgroup.max(), rYs >= Y_subgroup.min())
    wd = wasserstein_distance(rYs[mask],Ys[mask])
    n_subgroup = Y_subgroup.shape[0]
    return wd *(n_subgroup/Y.shape[0])**a

def compute_ed(Y, Y_subgroup,a=1/2):
    density_y_subgroup = gaussian_kde(Y_subgroup.T)
    density_y = gaussian_kde(Y.T)
    rYs = density_y_subgroup.resample(size=(20000,))
    Ys = density_y.resample(size=(20000,))
    rYs = rYs.reshape((20000,))
    Ys = Ys.reshape((20000,))
    #weights_sub = (rYs < Y_subgroup.max()) + (rYs > Y_subgroup.min())
    #weights = (Ys < Y_subgroup.max()) + (Ys > Y_subgroup.min())
    mask  = np.logical_and(rYs <= Y_subgroup.max(), rYs >= Y_subgroup.min())
    ed = energy_distance(rYs[mask],Ys[mask])
    n_subgroup = Y_subgroup.shape[0]
    return ed *(n_subgroup/Y.shape[0])**a

def _reconstruct_subgroup_from_rule(df, rule):
    op_map = {
        '<=': operator.le,
        '>=': operator.ge,
        '<': operator.lt,
        '>': operator.gt,
        '!=': operator.ne,
        '==': operator.eq,
        '=': operator.eq,
    }
    chain_pattern = re.compile(r"(-?[\d\.]+)\s*(<=|<|>=|>)\s*([\w][\w\-_.]*)\s*(<=|<|>=|>)\s*(-?[\d\.]+)")
    selection = np.ones(len(df), dtype=bool)
    for predicate in rule.split(" AND "):
        predicate = predicate.strip()
        match = chain_pattern.fullmatch(predicate)
        if match:
            left_val = float(match.group(1))
            left_op = match.group(2)
            field = match.group(3)
            right_op = match.group(4)
            right_val = float(match.group(5))
            left_sel = op_map[left_op](left_val, df[field])
            right_sel = op_map[right_op](df[field], right_val)
            selection &= left_sel & right_sel
            continue
        # Single comparison or other cases
        for op_str in op_map.keys():
            if op_str in predicate:
                predicate = predicate.replace(" ", "")
                field, value = map(str.strip, predicate.split(f"{op_str}"))
                op_func = op_map[op_str]
                selection &= op_func(df[field], float(value))
                break
        else:
            if predicate.startswith("NOT "):
                field = predicate[4:].strip()
                selection &= (df[field] == 0)
            else:
                field = predicate
                selection &= (df[field] == 1)
    return np.array(selection)

def reconstruct_subgroups_from_csv(data, results):
    subgroups = []
    rules = []
    for i, row in results.iterrows():
        rule = row["rule"]
        df = pd.DataFrame(data["data"], columns=data["feature_names"])
        subgroup = _reconstruct_subgroup_from_rule(df, rule)
        rule = rule.replace("AND", "$\\wedge$")
        rule = rule.replace("NOT ", "$\\neg$")
        rule = rule.replace("<=", "$\\leq$")
        rule = rule.replace(">=", "$\\geq$")
        rule = rule.replace("<", "$<$")
        rule = rule.replace(">", "$>$")
        rule = rule.replace("==", "$=$")
        rule = rule.replace("!=", "$\\neq$")
        rule = rule.replace("=", "$=$")
        subgroups.append(subgroup)
        rules.append(rule)
    return subgroups, rules