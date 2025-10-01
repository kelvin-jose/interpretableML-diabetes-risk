import re
import yaml
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_hybrid_models():
    with open('config.yml', 'r') as f:
        config = yaml.safe_load(f)

    # --- Load Data ---
    logging.info("Loading processed training data.")
    train_df = pd.read_csv(config['data']['processed_train_path'])
    train_df = train_df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    X_train = train_df.drop('readmitted', axis=1)
    y_train = train_df['readmitted']

if __name__ == '__main__':
    train_hybrid_models()