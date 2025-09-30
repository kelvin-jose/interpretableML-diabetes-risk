import os
import yaml
import json
import pickle
import logging
import argparse
import pandas as pd
import collections
if not hasattr(collections, 'Iterable'):
    import collections.abc
    collections.Iterable = collections.abc.Iterable
import six
import sys
sys.modules['sklearn.externals.six'] = six
from skrules import SkopeRules

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_model(constraints_path=None):
    with open('config.yml', 'r') as f:
        config = yaml.safe_load(f)
    
    train_df = pd.read_csv(config['data']['processed_train_path'])
    X_train = train_df.drop('readmitted', axis=1)
    y_train = train_df['readmitted']
    
    # Handle constraints from interactive session
    if constraints_path and os.path.exists(constraints_path):
        logging.info(f"Loading constraints from {constraints_path}")
        with open(constraints_path, 'r') as f:
            constraints = json.load(f)
        features_to_drop = constraints.get('features_to_drop', [])
        X_train = X_train.drop(columns=features_to_drop, errors='ignore')
        logging.info(f"Dropped constrained features: {features_to_drop}")
        model_path = config['models']['refined_model_path']
    else:
        logging.info("No constraints found. Training initial model.")
        model_path = config['models']['interpretable_model_path']

    skope_params = config['params']['skope_rules']
    model = SkopeRules(
        random_state=config['params']['random_state'],
        n_estimators=skope_params['n_estimators'],
        max_samples=skope_params['max_samples'],
        max_depth_duplication=skope_params['max_depth_duplication']
    )
    
    logging.info("Training SkopeRules model...")
    model.fit(X_train, y_train)
    logging.info("Training complete.")

    model_path = config['models']['interpretable_model_path']
    # Save the model
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    logging.info(f"Model saved to {model_path}")

    # Save the rules to a text file for inspection
    rules_report_path = model_path.replace('.pkl', '_rules.txt')
    with open(rules_report_path, 'w') as f:
        f.write("Top predictive rules for readmission:\n")
        f.write("="*40 + "\n")
        for rule in model.rules_:
            f.write(f"{rule}\n")
    logging.info(f"Model rules saved to {rules_report_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--refined', action='store_true', help="Train the refined model using constraints.")
    args = parser.parse_args()

    if args.refined:
        with open('config.yml', 'r') as f:
            config = yaml.safe_load(f)
        train_model(constraints_path=config['models']['constraints_path'])
    else:
        train_model()