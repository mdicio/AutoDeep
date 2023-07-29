import yaml
import numpy as np
from hyperopt import STATUS_OK, Trials, fmin, hp, space_eval, tpe
from hyperopt.pyll import scope


with open('./configuration/experiment_config.yml', 'r') as f:
    config = yaml.safe_load(f)

runs = config['runs']
param_grid = runs[3]["param_grid"]
param_grid["outer_params"]
def create_tabnet_search_space(param_grid):
    search_space = {}

    for param, values in param_grid.items():
        if param == "batch_size":
            continue
        
        if param == "virtual_batch_size_ratio":
            batch_size_space = hp.quniform("batch_size", min(param_grid["batch_size"]), max(param_grid["batch_size"]), 1)
            virtual_batch_size_ratio_space = hp.choice("virtual_batch_size_ratio", values)
            search_space["virtual_batch_size"] = scope.int(scope.mul(batch_size_space, virtual_batch_size_ratio_space))
            search_space["batch_size"] = batch_size_space
        elif isinstance(values[0], dict):
            param_space = []
            for v in values:
                inner_space = {}
                for k, v2 in v.items():
                    if isinstance(v2, list) and isinstance(v2[0], int):
                        inner_space[k] = scope.int(hp.quniform(k, min(v2), max(v2), 1))
                    elif isinstance(v2, list) and isinstance(v2[0], float):
                        inner_space[k] = scope.float(hp.loguniform(k, np.log(min(v2)), np.log(max(v2))))
                    elif isinstance(v2, list) and isinstance(v2[0], str):
                        inner_space[k] = hp.choice(k, v2)
                param_space.append({k: v2 for k, v2 in zip(v.keys(), inner_space)})
            search_space[param] =hp.choice(param, param_space)

        elif isinstance(values[0], str):
            search_space[param] = hp.choice(param, values)
        elif isinstance(values[0], int):
            search_space[param] = scope.int(hp.quniform(param, min(values), max(values), 1))
        elif isinstance(values[0], float):
            search_space[param] = scope.float(hp.loguniform(param, np.log(min(values)), np.log(max(values))))
    return search_space

sp = create_tabnet_search_space(param_grid)

