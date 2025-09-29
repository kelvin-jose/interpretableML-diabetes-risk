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

    df = df.drop(['weight', 'payer_code', 'medical_specialty'], axis=1)
    df = df.drop(['patient_nbr', 'encounter_id'], axis=1)

    # imputation
    df['race'] = df['race'].fillna(df['race'].mode()[0])

    # one-hot encoding categorical features
    categorical_cols = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    logging.info("Preprocessing complete.")
if __name__ == '__main__':
    preprocess_data()