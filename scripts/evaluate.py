import os
import yaml
import json
import pickle
import logging
import pandas as pd
import collections
if not hasattr(collections, 'Iterable'):
    import collections.abc
    collections.Iterable = collections.abc.Iterable
import six
import sys
sys.modules['sklearn.externals.six'] = six
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import export_text
from sklearn.metrics import classification_report

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

    for name, path in model_paths.items():
        logging.info(f"Evaluating model: {name}")
        results += f"--- {name} ---\n"
        
        try:
            with open(path, 'rb') as f:
                model = pickle.load(f)
            
            X_test_current = X_test.copy()
            # special handling for the refined model, which was trained on fewer features
            if "Human-Refined" in name and features_to_drop:
                X_test_current = X_test_current.drop(columns=features_to_drop, errors='ignore')

            # ensure feature names match for tree-based models
            if hasattr(model, 'feature_names_in_'):
                 X_test_current = X_test_current[model.feature_names_in_]

            predictions = model.predict(X_test_current)
            report = classification_report(y_test, predictions, target_names=['Not Readmitted', 'Readmitted <30d'])
            
            results += report + "\n\n"
            logging.info(f"Evaluation complete for {name}.")

        except FileNotFoundError:
            logging.warning(f"Model file not found at {path}. Skipping evaluation.")
            results += "Model not found. Please ensure it has been trained.\n\n"
        except Exception as e:
            logging.error(f"Could not evaluate model {name}. Error: {e}")
            results += f"An error occurred during evaluation: {e}\n\n"

    report_path = 'reports/results.txt'
    with open(report_path, 'w') as f:
        f.write(results)
    logging.info(f"Full evaluation report saved to {report_path}")
    print("\n" + results)

    generate_interpretability_artifacts(config)


def generate_interpretability_artifacts(config):
    os.makedirs('reports/figures', exist_ok=True)
    # 1. export hybrid decision tree rules
    try:
        with open(config['models']['hybrid_model_path'], 'rb') as f:
            model = pickle.load(f)
        
        with open(config['data']['processed_train_path'], 'rb') as f:
            feature_names = pd.read_csv(f, nrows=0).drop('readmitted', axis=1).columns.tolist()

        tree_rules = export_text(model, feature_names=feature_names)
        rules_path = 'reports/figures/decision_tree_rules.txt'
        with open(rules_path, 'w') as f:
            f.write("--- Rules for Hybrid Surrogate Decision Tree ---\n\n")
            f.write(tree_rules)
        logging.info(f"Decision tree rules saved to {rules_path}")

    except FileNotFoundError:
        logging.warning("Hybrid model not found, skipping rule export.")
    except Exception as e:
        logging.error(f"Could not export tree rules. Error: {e}")

    # 2. plot teacher model feature importance
    try:
        with open(config['models']['teacher_model_path'], 'rb') as f:
            model = pickle.load(f)
        
        feature_importances = pd.Series(model.feature_importances_, index=model.feature_name_).sort_values(ascending=False)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x=feature_importances.head(20), y=feature_importances.head(20).index)
        plt.title('Top 20 Feature Importances from LightGBM (Teacher Model)')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        
        importance_path = 'reports/figures/feature_importance.png'
        plt.savefig(importance_path)
        logging.info(f"Feature importance plot saved to {importance_path}")

    except FileNotFoundError:
        logging.warning("Teacher model not found, skipping feature importance plot.")
    except Exception as e:
        logging.error(f"Could not plot feature importances. Error: {e}")


if __name__ == '__main__':
    evaluate_models()