import os
import logging.config
import yaml


def setup_logging(path="logging.yaml"):
    """
    Setup logging configration
    :param path:
    :return:
    """
    if os.path.exists(path):
        with open(path, "rt") as f:
            config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)