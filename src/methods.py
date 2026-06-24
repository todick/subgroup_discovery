import torch
import pysubgroup as ps
import pandas as pd
import numpy as np
import warnings

from collections import namedtuple
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor, _tree
from src.syflow import *
from src.rsd.rulelist_class import MDLRuleList, reduce

warnings.filterwarnings("ignore")
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
verbose = False

def run_method(method, X, Y, config, n_subgroups, feature_names):
    subgroups = 123
    rules = 123
    # Xu, S. and Walter, N.P. and Kalofolias, J. and Vreeken, J. Learning Exceptional Subgroups by End-to-End Maximizing KL-divergence.
    if method == "syflow":
        subgroups, rules = run_syflow(X,Y,config,n_subgroups,feature_names,progressbar=False)
    # Lemmerich, F. and Becker, M. pysubgroup: Easy-to-use subgroup discovery in python.
    elif method == "beam-mean":
        subgroups, rules = run_beam_mean(X,Y,config,n_subgroups,feature_names)
    elif method == "beam-kl":
        subgroups, rules = run_beam_kl(X,Y,config,n_subgroups,feature_names)
    elif method == "dfs-mean":
        subgroups, rules = run_dfs_mean(X,Y,config,n_subgroups,feature_names)
    # Proença, H. M., Grünwald, P., Bäck, T., van Leeuwen, M. Robust Subgroup Discovery
    elif method == "rsd":
        subgroups, rules = run_rsd(X,Y,config,n_subgroups,feature_names)
    elif method == "cart":
        subgroups, rules = run_cart(X,Y,config,n_subgroups,feature_names)
    return subgroups, rules

def run_syflow(X, Y, config, n_subgroups, feature_names, return_flows=False, return_kl_estimates=False, progressbar=True):
    cut_points = torch.zeros((X.shape[1],2), dtype=torch.float32)
    scaler_x = StandardScaler()
    X = scaler_x.fit_transform(X)
    scaler_y = StandardScaler()
    Y = scaler_y.fit_transform(Y)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    Y_tensor = torch.tensor(Y, dtype=torch.float32)
    
    subgroups, rules, kl_estimates, priors = [], [], [], []
    pop_flow = None
    for n in range(n_subgroups):
        print("Discovering Subgroup #{}".format(n+1))
        for i in range(X.shape[1]):
            cut_points[i,0] = torch.quantile(X_tensor[:,i],0).to(torch.float32)
            cut_points[i,1] = torch.quantile(X_tensor[:,i],1).to(torch.float32)
        cut_points = torch.sort(cut_points,dim=1)[0].to(torch.float32)
        classifier = And_Finder(cut_points,temperature=config.temperature,use_weights=config.use_weights,bin_deviation=config.bin_deviation)
        flows, classifier, kl_estimate = syflow(X_tensor,Y_tensor,classifier,flow_population=pop_flow,subgroup_priors=priors,
                                            pop_train_epochs=config.pop_train_epochs,subgroup_train_epochs=config.subgroup_train_epochs,final_fit_epochs=config.final_fit_epochs,
                                        device=device,verbose=False,lr_flow=config.lr_flow,alpha=config.alpha,early_stopping_patience=config.early_stopping_patience,
                                    lr_classifier=config.lr_classifier,lambd=config.lambd,scaler=scaler_x,config=config,progressbar=progressbar)
        pop_flow = flows[0]
        priors.append(flows[1])
        classifier = classifier.to(torch.device("cpu"))
        subgroup = torch.argmax(classifier(X_tensor.to(torch.float32)),dim=1).detach().numpy()==1
        subgroups.append(subgroup)
        kl_estimates.append(kl_estimate)
        rules.append(classifier.get_rules(cut_points,scaler=scaler_x,feature_names=feature_names,X=X))
    if return_flows:
        return subgroups, rules, pop_flow, scaler_y, classifier
    if return_kl_estimates:
        return subgroups, rules, kl_estimates
    return subgroups, rules

def run_dfs_mean(X, Y, config, n_subgroups, feature_names):
    X = pd.DataFrame(X)
    Y = pd.DataFrame(Y)
    Y.columns = ["Y"]
    X.columns = [f"X{i}" for i in range(X.shape[1])]
    data = pd.concat([X,Y],axis=1)
    target = ps.NumericTarget("Y")
    search_space = ps.create_selectors(data, ignore=["Y"], nbins=config.ncutpoints, intervals_only=False)
    task = ps.SubgroupDiscoveryTask(
        data, 
        target, 
        search_space, 
        result_set_size=n_subgroups, 
        depth=config.sd_depth, 
        qf=ps.StandardQFNumeric(config.alpha)
    )

    result = ps.DFSNumeric().execute(task)
    result.to_dataframe()
    subgroups = []
    rules = []
    for i in range(n_subgroups):
        result_string = str(result.to_dataframe().iloc[i]["subgroup"])
        rules.append(replace_feature_names(result_string,feature_names))
        parts = result_string.split(" AND ")
        conditions = []
        for part in parts:
            if "==" in part:
                var, val = part.split("==")
                var = int(var[1:])
                val = convert(data,var, val)
                conditions.append((var,val,val))
                continue
            elif "<" in part:
                var, high = part.split("<")
                low = - np.infty
                var = int(var[1:])
                high = float(high)
            else:
                var, low = part.split(">=")
                high = np.infty
                var = int(var[1:])
                low = float(low)
            conditions.append((var,low,high))
            
        subgroup_member = np.ones((X.shape[0],),dtype=bool)
        for cond in conditions:
            var, low, high = cond
            var = int(var)
            subgroup_member = np.logical_and(subgroup_member, np.logical_and(X.iloc[:,var]>=low, X.iloc[:,var]<=high))
        subgroups.append(subgroup_member)
    return subgroups, rules  

def run_rsd(X,Y,config,n_subgroups,feature_names):
    scaler_y = StandardScaler()
    Y = scaler_y.fit_transform(Y)

    X = pd.DataFrame(X)
    X.columns = feature_names
    Y = pd.DataFrame(Y)
    Y.columns = ["Y"]
    target_model = "gaussian"
    task = "discovery"
    model = MDLRuleList(task = task, target_model = target_model,max_rules=n_subgroups, n_cutpoints=config.ncutpoints, max_depth=config.sd_depth, min_support=config.rsd_min_support)
    model.fit(X, Y)
    
    subgroups = []
    for subgroup in model._rulelist.subgroups:
        subgroup_member = reduce(lambda x,y: x & y, [item.activation_function(X).values for item in subgroup.pattern])
        subgroups.append(subgroup_member)
    rules = model._rulelist.get_rules()
    rules = [rule.split(" THEN ")[0].replace("\n", "").strip() for rule in rules]
    return subgroups, rules

def run_beam_mean(X,Y,config,n_subgroups,feature_names):
    X = pd.DataFrame(X)
    Y = pd.DataFrame(Y)
    Y.columns = ["Y"]
    X.columns = [f"X{i}" for i in range(X.shape[1])]
    data = pd.concat([X,Y],axis=1)
    target = ps.NumericTarget("Y")
    search_space = ps.create_selectors(data, ignore=["Y"],nbins=config.ncutpoints, intervals_only=False)
    task = ps.SubgroupDiscoveryTask(
        data, 
        target, 
        search_space, 
        result_set_size=n_subgroups, 
        depth=config.sd_depth, 
        qf=ps.StandardQFNumeric(config.alpha))

    result = ps.BeamSearch(beam_width=config.beam_width,beam_width_adaptive=False).execute(task)
    result.to_dataframe()
    subgroups = []
    rules = []
    for i in range(n_subgroups):
        result_string = str(result.to_dataframe().iloc[i]["subgroup"])
        rules.append(replace_feature_names(result_string,feature_names))
        parts = result_string.split(" AND ")
        conditions = []
        for part in parts:
            if "==" in part:
                var, val = part.split("==")
                var = int(var[1:])
                val = convert(data,var, val)
                conditions.append((var,val,val))
                continue
            elif "<" in part:
                var, high = part.split("<")
                low = - np.infty
                var = int(var[1:])
                high = float(high)
            else:
                var, low = part.split(">=")
                high = np.infty
                var = int(var[1:])
                low = float(low)
            conditions.append((var,low,high))
            
        subgroup_member = np.ones((X.shape[0],),dtype=bool)
        for cond in conditions:
            var, low, high = cond
            var = int(var)
            subgroup_member = np.logical_and(subgroup_member, np.logical_and(X.iloc[:,var]>=low, X.iloc[:,var]<=high))
        subgroups.append(subgroup_member)
    return subgroups, rules

def convert(df,var, val):
     if val.replace(".", "").isnumeric():
          return float(val)
     else:
          if str(df.dtypes[var]) == 'bool':
               return float(val=='True')
          
def run_beam_kl(X,Y,config,n_subgroups,feature_names):
    X = pd.DataFrame(X)
    Y = pd.DataFrame(Y)
    Y.columns = ["Y"]
    X.columns = [f"X{i}" for i in range(X.shape[1])]
    data = pd.concat([X,Y],axis=1)
    
    target = ps.NumericTarget("Y")
    search_space = ps.create_selectors(data, ignore=["Y"],nbins=config.ncutpoints, intervals_only=False)
    task = ps.SubgroupDiscoveryTask(
        data, 
        target, 
        search_space, 
        result_set_size=n_subgroups, 
        depth=config.sd_depth, 
        qf=QF_WKL(config.alpha))

    result = ps.BeamSearch(beam_width=config.beam_width,beam_width_adaptive=False).execute(task)
    result.to_dataframe()
    subgroups = []
    rules = []
    for i in range(n_subgroups):
        result_string = str(result.to_dataframe().iloc[i]["subgroup"])
        rules.append(replace_feature_names(result_string,feature_names))
        parts = result_string.split(" AND ")
        conditions = []
        for part in parts:
            if "==" in part:
                var, val = part.split("==")
                var = int(var[1:])
                val = val = convert(data,var, val)
                conditions.append((var,val,val))
                continue
            elif "<" in part:
                var, high = part.split("<")
                low = - np.infty
                var = int(var[1:])
                high = float(high)
            else:
                var, low = part.split(">=")
                high = np.infty
                var = int(var[1:])
                low = float(low)
            conditions.append((var,low,high))
            
        subgroup_member = np.ones((X.shape[0],),dtype=bool)
        for cond in conditions:
            var, low, high = cond
            var = int(var)
            subgroup_member = np.logical_and(subgroup_member, np.logical_and(X.iloc[:,var]>=low, X.iloc[:,var]<=high))
        subgroups.append(subgroup_member)
    return subgroups, rules

def run_cart(X, Y, config, n_subgroups, feature_names):
    max_depth = config.sd_depth
    clf = DecisionTreeRegressor(max_depth=max_depth)
    clf.fit(X, Y)
    tree = clf.tree_
    subgroups = []
    rules = []

    def traverse(node, depth, samples_mask, prev_rules):
        nonlocal subgroups, rules
        if depth < max_depth:
            if tree.feature[node] != _tree.TREE_UNDEFINED:
                left_child = tree.children_left[node]
                right_child = tree.children_right[node]
                left_mask = samples_mask & (X[:, tree.feature[node]] <= tree.threshold[node])
                right_mask = samples_mask & (X[:, tree.feature[node]] > tree.threshold[node])
                # if feature only has zeroes and ones, make the rule more readable
                nel = np.round(np.unique(X[:, tree.feature[node]]))
                if len(nel) == 2 and 0 in nel and 1 in nel:
                    left_rules = prev_rules + [f"NOT {feature_names[tree.feature[node]]}"]
                    right_rules = prev_rules + [f"{feature_names[tree.feature[node]]}"]
                else:
                    left_rules = prev_rules + [f"{feature_names[tree.feature[node]]} <= {tree.threshold[node]}"]
                    right_rules = prev_rules + [f"{feature_names[tree.feature[node]]} > {tree.threshold[node]}"]
                subgroups += [left_mask, right_mask]
                rules += [left_rules, right_rules]
                traverse(left_child, depth + 1, left_mask, left_rules)
                traverse(right_child, depth + 1, right_mask, right_rules)

    traverse(0, 0, np.ones(X.shape[0], dtype=bool), [])
    rules = [" AND ".join(rule) for rule in rules]

    # select the n_subgroups with the highest qf_wkl
    if len(subgroups) > n_subgroups:
        X = pd.DataFrame(X)
        Y = pd.DataFrame(Y)
        Y.columns = ["Y"]
        X.columns = [f"X{i}" for i in range(X.shape[1])]
        data = pd.concat([X,Y],axis=1)
        qf_wkl = QF_WKL(config.alpha)
        qf_wkl.calculate_constant_statistics(data, ps.NumericTarget("Y"))
        subgroup_scores = []
        for i, subgroup in enumerate(subgroups):
            statistics = qf_wkl.calculate_statistics(subgroup, ps.NumericTarget("Y"), data)
            score = qf_wkl.evaluate(subgroup, ps.NumericTarget("Y"), data, statistics)
            subgroup_scores.append((score, i))
        subgroup_scores.sort(reverse=True, key=lambda x: x[0])
        selected_indices = [idx for _, idx in subgroup_scores[:n_subgroups]]
        subgroups = [subgroups[i] for i in selected_indices]
        rules = [rules[i] for i in selected_indices]

    return subgroups, rules


# KL divergence quality measure assuming normal distribution

class QF_WKL(ps.BoundedInterestingnessMeasure):
    tpl = namedtuple('StandardQFNumeric_parameters', ('size_sg', 'mean', "std", 'estimate'))

    def __init__(self, a, invert=False, estimator='sum'):
        self.a = a
        self.invert = invert
        self.required_stat_attrs = ('size_sg', 'mean')
        self.dataset_statistics = None
        self.all_target_values = None
        self.has_constant_statistics = False

    def calculate_constant_statistics(self, data, target):
        self.all_target_values = data[target.target_variable].to_numpy()
        target_mean = np.mean(self.all_target_values)
        data_size = len(data)
        std = np.std(self.all_target_values)
        self.dataset_statistics = QF_WKL.tpl(data_size, target_mean, std,None)
        self.has_constant_statistics = True

    def evaluate(self, subgroup, target, data, statistics=None):
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        dataset = self.dataset_statistics
        size_sg = statistics.size_sg
        mean_sg = statistics.mean
        std_sg = statistics.std + 0.0000001
        mean_dataset = dataset.mean
        std_dataset = dataset.std
        if size_sg < 2:
            return 0
        kl = np.log2(std_dataset/std_sg) + (std_sg**2+(mean_sg-mean_dataset)**2)/(2*std_dataset**2)
        w = size_sg/dataset.size_sg
        return w**(self.a)*kl

    def calculate_statistics(self, subgroup, target, data, statistics=None):
        cover_arr, sg_size = ps.get_cover_array_and_size(subgroup, len(self.all_target_values), data)
        sg_mean = 0
        sg_target_values = 0
        sg_std = 0
        if sg_size > 1:
            sg_target_values = self.all_target_values[cover_arr]
            sg_mean = np.mean(sg_target_values)
            sg_std = np.std(sg_target_values)
        return QF_WKL.tpl(sg_size, sg_mean, sg_std, None)
    
def replace_feature_names(rule,feature_names):
    # replace X0<=... with feature_names[0]<=...
    for i in reversed(range(len(feature_names))):
        if "X"+str(i) in rule:
            rule = rule.replace("X"+str(i),feature_names[i])
    return rule

