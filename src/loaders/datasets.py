import src.loaders.distance_based as distance_based
import src.loaders.workload as workload

datasets = {
    "distance_based": distance_based,
    "workload": workload
}

def load(dataset, casestudy, kwargs={}):
    return datasets[dataset].loaders[casestudy](**kwargs)

def get_casestudy_names(dataset):
    return datasets[dataset].loaders.keys()