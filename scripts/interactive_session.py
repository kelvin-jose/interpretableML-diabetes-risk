import yaml
import json
import pickle
import logging
import collections
if not hasattr(collections, 'Iterable'):
    import collections.abc
    collections.Iterable = collections.abc.Iterable
import six
import sys
sys.modules['sklearn.externals.six'] = six

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def start_interactive_session():
    with open('config.yml', 'r') as f:
        config = yaml.safe_load(f)
    
    model_path = config['models']['interpretable_model_path']
    constraints_path = config['models']['constraints_path']

    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    except FileNotFoundError:
        logging.error("Initial model not found. Please run `train_interpretable.py` first.")
        return
    
    print("\n--- Interactive Model Refinement Session ---")
    print("You will be shown the top predictive rules. Please provide feedback.")
    print("Based on your feedback, we will generate constraints for retraining.\n")

    rules = model.rules_
    features_to_drop = set()

    for i, (rule, precision_recall) in enumerate(rules[:10]): # Review top 10 rules
        print(f"Rule {i+1}: {rule}")
        print(f"   Precision: {precision_recall[0]:.2f}, Recall: {precision_recall[1]:.2f}")
        
        feedback = input("Is this rule clinically irrelevant or based on a proxy feature? (y/n): ").lower()
        if feedback == 'y':
            print("This rule will be de-prioritized by constraining its features.")
            # Simple constraint: find features in the rule and add them to a drop list
            rule_features = [feat for feat in model.feature_names_ if feat in rule]
            for feat in rule_features:
                features_to_drop.add(feat)
            print(f"   -> Added {rule_features} to drop list.\n")
    
    constraints = {'features_to_drop': list(features_to_drop)}
    with open(constraints_path, 'w') as f:
        json.dump(constraints, f, indent=4)

    logging.info(f"Constraints saved to {constraints_path}")
    print("\n--- Session Complete ---")
    print(f"To retrain the model with these new constraints, run:")
    print("`python scripts/train_interpretable.py --refined`")


if __name__ == '__main__':
    start_interactive_session()