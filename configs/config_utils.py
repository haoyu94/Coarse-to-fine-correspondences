import os
import yaml

def load_config(path):
    '''
    Load config file
    :param path: path to config file
    :return: a dict of configuration parameters, merge sub_dicts
    '''
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)

    config = dict()
    for key, value in cfg.items():
        for k, v in value.items():
            config[k] = v
    return config