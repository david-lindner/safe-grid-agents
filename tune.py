from main import main

import ray
import ray.tune as tune
import yaml
import argparse

def replace_hparams(config_dict):
    for k, v in config_dict.items():
        try:
            if k == 'config':
                for j in v.keys():
                    v[j] = eval(v[j])
                    config_dict[k][j] = v[j]
            else:
                replace_hparams(v)
        except AttributeError:
            continue
    return config_dict

# Initialize Ray Tune
ray.init()
tune.register_trainable("main", main)

# Read config
parser = argparse.ArgumentParser(description='Tuning (Hopefully) Safe Agents in Gridworlds')
parser.add_argument('--experiment', type=str, required=True, help="Experimental configuration(s) YAML")
parsed = parser.parse_args()
with open(parsed.experiment) as exp_yaml:
    exp_configs = yaml.load(exp_yaml)
    experiments = replace_hparams(exp_configs)

# Run Ray Tune
all_trials = tune.run_experiments(experiments)
