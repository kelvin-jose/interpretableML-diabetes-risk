import os
import io
import yaml
import requests
import zipfile
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
    try:
        logging.info(f"Downloading data from {url}...")
        r = requests.get(url)
        r.raise_for_status()
        z = zipfile.ZipFile(io.BytesIO(r.content))
        
        # The zip file contains 'dataset_diabetes/diabetic_data.csv'
        # Extract it and rename it to the path specified in config
        with z.open('dataset_diabetes/diabetic_data.csv') as source, open(raw_path, 'wb') as target:
            target.write(source.read())

        logging.info(f"Successfully downloaded and saved data to {raw_path}")

    except requests.exceptions.RequestException as e:
        logging.error(f"Error downloading data: {e}")
    except zipfile.BadZipFile:
        logging.error("Error: Downloaded file is not a valid zip file.")

if __name__ == '__main__':
    fetch_data()