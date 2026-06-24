import pandas as pd 
from src.loaders.generic import enum_to_numeric

datapath = "./data/workload"

def get_workload_names(name):
    df = pd.read_csv(datapath + f"/{name}/measurements.csv")
    workload = df['workload'].unique().tolist()
    return workload

def load(name, target, workload=None):
    sample = pd.read_csv(datapath + f"/{name}/sample.csv")
    for col in sample.columns:
        if sample[col].dtype == bool:
            sample[col] = sample[col].astype(int)
    measurements = pd.read_csv(datapath + f"/{name}/measurements.csv")
    measurements = measurements[[target, 'workload', 'config_id']]

    if workload is None or type(workload) == int:
        workloads = measurements['workload'].unique().tolist()
        if workload is not None:
            workloads = [workloads[workload]]
    else:
        workloads = [workload]    
    
    df = pd.merge(sample, measurements, on='config_id')
    df = df.loc[:,df.nunique() > 1]
    df = df.drop('config_id', axis=1)
    df = df[df['workload'].isin(workloads)]
    output = {}
    df["workload"] = enum_to_numeric(df["workload"])
    output["target"] = df[target].to_numpy()
    output["feature_names"] = df.drop(target, axis=1).columns.values.tolist()
    output["data"] = df.drop(target, axis=1).to_numpy()
    output["target_name"] = target
    output["workloads"] = workloads
    output["df"] = df
    return output

selected_workloads = {
    "batik": "karte",
    "dconvert": "psd-large",
    "h2": "voter-2",
    "jump3r": "helix.wav",
    "kanzi": "ambivert",
    "lrzip": "vmlinux-5.10.tar",
    "x264": "Netflix_Crosswalk_4096x2160_60fps_10bit_420_short.y4m",
    "xz": "fannie_mae_500k.tar",
    "z3": "LRA_formula_277.smt2"
}

loaders = {
    "batik": lambda workload=selected_workloads["batik"]: load("batik", "time", workload),
    "dconvert": lambda workload=selected_workloads["dconvert"]: load("dconvert", "time", workload),
    "h2": lambda workload=selected_workloads["h2"]: load("h2", "throughput", workload),
    "jump3r": lambda workload=selected_workloads["jump3r"]: load("jump3r", "time", workload),
    "kanzi": lambda workload=selected_workloads["kanzi"]: load("kanzi", "time", workload),
    "lrzip": lambda workload=selected_workloads["lrzip"]: load("lrzip", "time", workload),
    "x264": lambda workload=selected_workloads["x264"]: load("x264", "time", workload),
    "xz": lambda workload=selected_workloads["xz"]: load("xz", "time", workload),
    "z3": lambda workload=selected_workloads["z3"]: load("z3", "time", workload)
}