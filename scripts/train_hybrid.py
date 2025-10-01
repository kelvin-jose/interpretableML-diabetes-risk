import re
import os
import yaml
import pickle
import logging
import pandas as pd
import lightgbm as lgb
from sklearn.tree import DecisionTreeClassifier

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

    os.makedirs(os.path.dirname(config['models']['teacher_model_path']), exist_ok=True)

    # --- Train the Teacher Model (LightGBM) ---
    logging.info("Training the 'teacher' model (LightGBM)...")
    lgbm_params = config['params']['lightgbm']
    teacher_model = lgb.LGBMClassifier(
        random_state=config['params']['random_state'],
        **lgbm_params
    )
    
    teacher_model.fit(X_train, y_train)
    logging.info("Teacher model training complete.")

    teacher_model_path = config['models']['teacher_model_path']
    with open(teacher_model_path, 'wb') as f:
        pickle.dump(teacher_model, f)
    logging.info(f"Teacher model saved to {teacher_model_path}")

    # Train the Student Model (Surrogate Decision Tree) ---
    logging.info("Training the 'student' surrogate model (Decision Tree)...")
    
    # Generate predictions from the teacher model to be used as labels for the student
    teacher_predictions = teacher_model.predict(X_train)
    
    dt_params = config['params']['decision_tree']
    student_model = DecisionTreeClassifier(
        random_state=config['params']['random_state'],
        **dt_params
    )
    
    # The student learns to mimic the teacher's predictions
    student_model.fit(X_train, teacher_predictions)
    logging.info("Student surrogate model training complete.")

    student_model_path = config['models']['hybrid_model_path']
    with open(student_model_path, 'wb') as f:
        pickle.dump(student_model, f)
    logging.info(f"Student surrogate model saved to {student_model_path}")

if __name__ == '__main__':
    train_hybrid_models()