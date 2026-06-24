
import pandas as pd 
from src.loaders.generic import xor_to_numeric

datapath = "./data/distance_based"

def load_7z(target='Performance'):
    targets = ['Performance', 'Size']
    df = pd.read_csv(datapath + "/7z/measurements.csv", sep=';')
    output = {}
    df = xor_to_numeric(df, 'BlockSize', {f'BlockSize_{i}': i for i in [2**n for n in range(0, 13)]})
    df = xor_to_numeric(df, 'x', {f'x_{i}': i for i in range(0, 11, 2)})   
    df = xor_to_numeric(df, 'Files', {f'Files_{i}': i for i in range(0, 101, 10)})
    df = df.loc[:,df.nunique() > 1]
    output["target"] = df[target].to_numpy()
    output["feature_names"] = df.drop(targets, axis=1).columns.values.tolist()
    output["data"] = df.drop(targets, axis=1).to_numpy()
    output["target_name"] = target
    return output

def load_berkeleydbc():
    target = 'Performance'
    df = pd.read_csv(datapath + "/BerkeleyDBC/measurements.csv", sep=';')
    output = {}
    df = xor_to_numeric(df, 'CACHESIZE', {f'CS{i}MB': i for i in [16, 32, 64, 512]})
    df = xor_to_numeric(df, 'PAGESIZE', {f'PS{i}K': i for i in [1, 4, 8, 16, 32]})   
    df = df.loc[:,df.nunique() > 1]
    output["target"] = df[target].to_numpy()
    output["feature_names"] = df.drop(target, axis=1).columns.values.tolist()
    output["data"] = df.drop(target, axis=1).to_numpy()
    output["target_name"] = target
    return output

def load_dune():
    target = 'Performance'
    df = pd.read_csv(datapath + "/Dune/measurements.csv", sep=';')
    output = {}
    df = xor_to_numeric(df, 'PreSmoothing', {f'pre_{i}': i for i in range(0, 7)})   
    df = xor_to_numeric(df, 'PostSmooting', {f'post_{i}': i for i in range(0, 7)})
    df = xor_to_numeric(df, 'Cells', {f'pre_{i}': i for i in range(50, 56)})   
    df = df.loc[:,df.nunique() > 1]
    output["target"] = df[target].to_numpy()
    output["feature_names"] = df.drop(target, axis=1).columns.values.tolist()
    output["data"] = df.drop(target, axis=1).to_numpy()
    output["target_name"] = target
    return output

def load_hipacc():
    target = 'Performance'
    df = pd.read_csv(datapath + "/Hipacc/measurements.csv", sep=';')
    output = {}
    df = xor_to_numeric(df, 'padding', {f'padding_{i}': i for i in range(0, 513, 32)})
    df = xor_to_numeric(df, 'pixelPerThread', {f'pixelPerThread_{i}': i for i in range(1, 5)})
    df = df.loc[:,df.nunique() > 1]
    output["target"] = df[target].to_numpy()
    output["feature_names"] = df.drop(target, axis=1).columns.values.tolist()
    output["data"] = df.drop(target, axis=1).to_numpy()
    output["target_name"] = target
    return output

def load_javagc():
    target = 'GC Time'
    df = pd.read_csv(datapath + "/JavaGC/measurements.csv", sep=';')
    output = {}
    df = xor_to_numeric(df, 'MaxTenuringThreshold', {f'MaxTenuringThreshold_{i}': i for i in [5, 10, 15]})
    df = xor_to_numeric(df, 'MinSurvivorRatio', {f'MinSurvivorRatio_{i}': i for i in [1, 4, 7, 10]})
    df = xor_to_numeric(df, 'NewRatio', {f'NewRatio_{i}': i for i in [1, 2, 4, 8, 16, 32]})
    df = xor_to_numeric(df, 'SurvivorRatio', {f'SurvivorRatio_{i}': i for i in range(1, 32, 5)})
    df = xor_to_numeric(df, 'TenuredGenerationSizeSupplementDecay', {f'TenuredGenerationSizeSupplementDecay_{i}': i for i in [2, 4, 8, 16, 50, 70]})
    df = df.loc[:,df.nunique() > 1]
    output["target"] = df[target].to_numpy()
    output["feature_names"] = df.drop(target, axis=1).columns.values.tolist()
    output["data"] = df.drop(target, axis=1).to_numpy()
    output["target_name"] = target
    return output

def load_llvm():
    target = 'Performance'
    df = pd.read_csv(datapath + "/LLVM/measurements.csv", sep=';')
    output = {} 
    df = df.loc[:,df.nunique() > 1]
    output["target"] = df[target].to_numpy()
    output["feature_names"] = df.drop(target, axis=1).columns.values.tolist()
    output["data"] = df.drop(target, axis=1).to_numpy()
    output["target_name"] = target
    return output

def load_lrzip():
    target = 'Performance'
    df = pd.read_csv(datapath + "/lrzip/measurements.csv", sep=';')
    output = {}
    df = xor_to_numeric(df, 'level', {f'level{i}': i for i in range(1, 10)})
    df = df.loc[:,df.nunique() > 1]
    output["target"] = df[target].to_numpy()
    output["feature_names"] = df.drop(target, axis=1).columns.values.tolist()
    output["data"] = df.drop(target, axis=1).to_numpy()
    output["target_name"] = target
    return output

def load_polly(target='Performance'):
    targets = ['Performance', 'ElapsedTime']
    df = pd.read_csv(datapath + "/Polly/measurements.csv", sep=';')
    output = {}
    df = xor_to_numeric(df, 'pollyrtcmaxparameters', {f'pollyrtcmaxparameters_{i}': i for i in [1, 2, 4, 8, 16]})
    df = xor_to_numeric(df, 'pollyrtcmaxconstantterm', {f'pollyrtcmaxconstantterm_{i}': i for i in [1, 10, 100, 1000, 10000]})
    df = xor_to_numeric(df, 'pollyrtcmaxcoefficient', {f'pollyrtcmaxcoefficient_{i}': i for i in [1, 10, 100, 1000, 10000]})
    df = xor_to_numeric(df, 'pollydefaulttilesize', {f'pollydefaulttilesize_{i}': i for i in [4, 16, 64, 256, 1024]})
    df = df.loc[:,df.nunique() > 1]
    output["target"] = df[target].to_numpy()
    output["feature_names"] = df.drop(targets, axis=1).columns.values.tolist()
    output["data"] = df.drop(targets, axis=1).to_numpy()
    output["target_name"] = target
    return output

def load_vp9(target='Size'):
    targets = ['UserTime', 'ElapsedTime', 'Size', 'PSNR Avg', 'PSNR Ov']
    df = pd.read_csv(datapath + "/VP9/measurements.csv", sep=';')
    output = {}
    df = xor_to_numeric(df, 'cpuUsed', {f'cpuUsed_{i}': i for i in range(0, 9, 2)})
    df = xor_to_numeric(df, 'Threads', {f'Threads_{i}': i for i in range(2, 11, 2)})
    df = xor_to_numeric(df, 'TileColumns', {f'TileColumns_{i}': i for i in [0, 3, 6]})
    df = xor_to_numeric(df, 'bitRate', {f'bitRate_{i}': i for i in range(300, 1501, 300)})
    df = xor_to_numeric(df, 'lagInFrames', {f'lagInFrames_{i}': i for i in [0, 8, 16]})
    df = df.loc[:,df.nunique() > 1]
    output["target"] = df[target].to_numpy()
    output["feature_names"] = df.drop(targets, axis=1).columns.values.tolist()
    output["data"] = df.drop(targets, axis=1).to_numpy()
    output["target_name"] = target
    return output

def load_x264():
    target = 'Performance'
    df = pd.read_csv(datapath + "/x264/measurements.csv", sep=';')
    output = {}
    df = xor_to_numeric(df, 'rc_lookahead', {f'rc_lookahead_{i}': i for i in [20, 40, 60]})
    df = xor_to_numeric(df, 'ref', {f'ref_{i}': i for i in [1, 5, 9]})  
    df = df.loc[:,df.nunique() > 1]
    output["target"] = df[target].to_numpy()
    output["feature_names"] = df.drop(target, axis=1).columns.values.tolist()
    output["data"] = df.drop(target, axis=1).to_numpy()
    output["target_name"] = target
    return output

loaders = {
    "7z": load_7z,
    "BerkeleyDBC": load_berkeleydbc,
    "Dune": load_dune,
    "Hipacc": load_hipacc,
    "JavaGC": load_javagc,
    "LLVM": load_llvm,
    "LRZip": load_lrzip,
    "Polly": load_polly,
    "VP9": load_vp9,
    "x264": load_x264
}