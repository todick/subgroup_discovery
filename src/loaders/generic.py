import pandas as pd
datapath = "./data"

def xor_to_numeric(df, parent, mapping):
    cp = df.copy(True)
    cp[parent] = 0
    for k,v in mapping.items():
        if k in cp.columns:
            cp[parent] = cp[parent] ^ (cp[k]*v)
            cp.drop(k, axis=1, inplace=True)
    return cp

def enum_to_numeric(series):
    if series.dtype != 'object':
        return series
    values = series.unique()
    mapping = {}
    for ind,v in enumerate(values):
        mapping[v] = ind
    series = series.map(mapping)
    return series

def load_tuning_results():
    df = pd.read_csv(datapath + "/tuning_results/measurements.csv", sep=';')
    output = {}
    df = df.apply(enum_to_numeric)
    for col in df.columns:
        if len(df[col].unique()) == 1:
            df.drop(col, axis=1, inplace=True)
    output["df"] = df.copy(True)
    output["target"] = df["quality"].to_numpy()
    feature_names = df.columns.values.tolist()
    feature_names.remove("quality")
    output["feature_names"] = feature_names
    output["data"] = df.drop("quality",axis=1).to_numpy()
    output["target_name"] = "quality"
    return output

def load_synthetic(name):
    df = pd.read_csv(datapath + f"/synthetic/{name}/result1/new_measurements.csv", sep=';')
    output = {}
    output["df"] = df.copy(True)
    output["target"] = df["Performance"].to_numpy()
    feature_names = df.columns.values.tolist()
    feature_names.remove("Performance")
    output["feature_names"] = feature_names
    output["data"] = df.drop("Performance",axis=1).to_numpy()
    output["target_name"] = "Performance"
    return output

def load_simulation():
    df = pd.read_csv(datapath + "/simulation/samples.csv")
    output = {}
    output["df"] = df.copy(True)
    df.drop(['root'], axis=1, inplace=True)
    
    output["target"] = df["y"].to_numpy()
    output["feature_names"] = df.columns.values.tolist()
    output["feature_names"].remove("y")
    output["data"] = df.drop("y",axis=1).to_numpy()
    output["target_name"] = "Insurance Rate"
    return output

def load_generic_X_y(path, sep=';'):
    df = pd.read_csv(path, sep=sep)
    output = {}
    output["df"] = df.copy(True)
    df.drop(index=0, axis=1, inplace=True)

    output["target"] = df["Performance"].to_numpy()
    output["feature_names"] = df.columns.values.tolist()
    output["feature_names"].remove("Performance")
    output["data"] = df.drop("Performance",axis=1).to_numpy()
    output["target_name"] = "Target"
    return output    