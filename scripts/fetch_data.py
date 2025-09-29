import os
import yaml
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_data():
    with open('config.yml', 'r') as f:
        config = yaml.safe_load(f)

    url = config['data']['dataset_url']
    raw_path = config['data']['raw_path']
    raw_dir = os.path.dirname(raw_path)

    if not os.path.exists(raw_dir):
        os.makedirs(raw_dir)
        logging.info(f"Created directory: {raw_dir}")

    if os.path.exists(raw_path):
        logging.info("Data already exists. Skipping download.")
        return

if __name__ == '__main__':
    fetch_data()