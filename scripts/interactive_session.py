import yaml
import pickle
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def start_interactive_session():
    with open('config.yml', 'r') as f:
        config = yaml.safe_load(f)
    
    model_path = config['models']['interpretable_model_path']

    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    except FileNotFoundError:
        logging.error("Initial model not found. Please run `train_interpretable.py` first.")
        return

if __name__ == '__main__':
    start_interactive_session()