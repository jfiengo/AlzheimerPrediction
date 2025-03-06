import pandas as pd
import yaml

class Ingestion:
    def __init__(self):
        self.config = self.load_config()

    def load_config(self):
        with open("config.yml", "r") as file:
            return yaml.safe_load(file)

    def load_data(self):
        data_path = self.config['data']['path']
        data = pd.read_csv(data_path)
        return data