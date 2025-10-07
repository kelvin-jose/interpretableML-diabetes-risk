import yaml
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_models():
    with open('config.yml', 'r') as f:
        config = yaml.safe_load(f)

    # --- load test data ---
    logging.info("Loading test data for evaluation.")
    try:
        test_df = pd.read_csv(config['data']['processed_test_path'])
        X_test = test_df.drop('readmitted', axis=1)
        y_test = test_df['readmitted']
    except FileNotFoundError:
        logging.error("Test data not found. Please run preprocessing first.")
        return

if __name__ == '__main__':
    evaluate_models()