import yaml
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_model():
    with open('config.yml', 'r') as f:
        config = yaml.safe_load(f)
    
    train_df = pd.read_csv(config['data']['processed_train_path'])
    X_train = train_df.drop('readmitted', axis=1)
    y_train = train_df['readmitted']

if __name__ == '__main__':
    train_model()