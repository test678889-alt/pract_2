import pandas as pd
from sklearn.datasets import load_diabetes
import yaml
from logger import get_logger

def load_config():
    with open("config/config.yaml") as f:
        return yaml.safe_load()


def load_and_process_data():

    config = load_config()
    logger = get_logger(config["logging"]["log_file"])

    logger.info("Starting data ingestion")

    data = load_diabetes(as_frame=True)
    df =  data.frame

    df.to_csv(config['data']['raw_path'],index=False)
    df.to_csv(config['data']['processed_data'],index=False)

if __name__ == '__main__':
    load_and_process_data()
