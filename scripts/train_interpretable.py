import yaml
import logging
import pandas as pd
from skrules import SkopeRules

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_model():
    with open('config.yml', 'r') as f:
        config = yaml.safe_load(f)
    
    train_df = pd.read_csv(config['data']['processed_train_path'])
    X_train = train_df.drop('readmitted', axis=1)
    y_train = train_df['readmitted']

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

if __name__ == '__main__':
    train_model()