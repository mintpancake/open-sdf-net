import json
import os

CONFIG_PATH = "../configs/config.json"


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def read_config():
    with open(CONFIG_PATH) as json_file:
        config = json.load(json_file)
        # globals().update(config)
    return config


def update_config(config):
    with open(CONFIG_PATH, 'w') as json_file:
        json.dump(config, json_file, indent=4)
