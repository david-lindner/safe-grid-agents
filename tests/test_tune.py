# from main import main

# import ray
# import ray.tune as tune
import yaml
import argparse
import sys
sys.path.append('../')
from tuning import replace_hparams

def main(config, reporter):
    pass

class DummyTune(object):
    """Dummy Ray Tune Tester"""
    def __init__(self):
        super(DummyTune, self).__init__()
        self.container = []
    
    @classmethod
    def register_trainable(cls, name, func):
        assert eval(name) == func
        assert name == 'main'

    @classmethod
    def grid_search(self, grid_list):
        for elt in grid_list:
            self.container.append(elt)

    @classmethod
    def run_experiments(cls, experiments):
        for k, v in experiments.items():
            try:
                if k == 'config':
                    for j in v.keys():
                        print(v[j])
                else:
                    replace_hparams(v)
            except AttributeError:
                continue

# Read config
parser = argparse.ArgumentParser(description='Tuning (Hopefully) Safe Agents in Gridworlds')
parser.add_argument('--experiment', type=str, required=True, help="Experimental configuration(s) YAML")
parsed = parser.parse_args('--experiment fixtures/basic_tune.yaml'.split())

def test_yaml_to_tune():
    with open(parsed.experiment) as exp_yaml:
    exp_configs = yaml.load(exp_yaml)
    experiments = replace_hparams(exp_configs)

    # Run Ray Tune
    tune = DummyTune()

    # Initialize Ray Tune
    tune.register_trainable("main", main)
    all_trials = tune.run_experiments(experiments)

    # Checks
    assert len(tune.container) == 10
    assert set(tune.container) == set([.1, .2, .3, .4, .6])

test_yaml_to_tune()