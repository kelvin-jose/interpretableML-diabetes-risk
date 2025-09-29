import yaml
import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def preprocess_data():
    with open('config.yml', 'r') as f:
        config = yaml.safe_load(f)
    
    raw_path = config['data']['raw_path']
    logging.info(f"Loading data from {raw_path}")
    df = pd.read_csv(raw_path)

    df.replace('?', np.nan, inplace=True)

    # target variable
    df['readmitted'] = (df['readmitted'] == '<30').astype(int)

if __name__ == '__main__':
    preprocess_data()