import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

class Ingestion:
    def __init__(self):
        self.config = self.load_config()

    def load_config(self):
        with open("config.yml", "r") as file:
            return yaml.safe_load(file)

    def load_data(self):
        data_path = self.config['data']['path']
        test_size = self.config['train']['test_size']
        random_state = self.config['train']['random_state']
        data = pd.read_csv(data_path)
        train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state)
        return train_data, test_data