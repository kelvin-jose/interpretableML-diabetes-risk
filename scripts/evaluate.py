import os
import yaml
import json
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
    
    model_paths = {
        "Interpretable Baseline (SkopeRules)": config['models']['interpretable_model_path'],
        "Human-Refined Model (SkopeRules)": config['models']['refined_model_path'],
        "Black-Box Teacher (LightGBM)": config['models']['teacher_model_path'],
        "Hybrid Surrogate (Decision Tree)": config['models']['hybrid_model_path'],
    }
    
    constraints = {}
    constraints_path = config['models']['constraints_path']
    if os.path.exists(constraints_path):
        with open(constraints_path, 'r') as f:
            constraints = json.load(f)
    features_to_drop = constraints.get('features_to_drop', [])

if __name__ == '__main__':
    evaluate_models()