import os
import yaml
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

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

    # data splitting
    X = df.drop('readmitted', axis=1)
    y = df['readmitted']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config['params']['test_size'], 
        random_state=config['params']['random_state'],
        stratify=y
    )

    # Combine X and y for saving
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    processed_dir = os.path.dirname(config['data']['processed_train_path'])
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
    
    train_df.to_csv(config['data']['processed_train_path'], index=False)
    test_df.to_csv(config['data']['processed_test_path'], index=False)
    logging.info(f"Saved processed data to {processed_dir}")

if __name__ == '__main__':
    preprocess_data()