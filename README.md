# Human-Centered Interpretable Modeling for Diabetic Readmission Risk

This project explores the development of interpretable machine learning models for predicting 30-day hospital readmission for diabetic patients. It is designed to demonstrate an understanding of human-centered AI, pattern-based modeling, and hybrid approaches that balance interpretability and predictive performance, aligning with research in Human-Centered Interpretable ML.

## Project Overview

The core idea is to move beyond black-box models and create a workflow where a domain expert can interact with, guide, and trust the model's predictions.

1.  **Interpretable Baseline:** We first train a rule-based model (`skope-rules`) that generates simple, logical IF-THEN rules.
2.  **Human-in-the-Loop:** An interactive script allows a user to review these rules, provide feedback on their clinical relevance, and generate constraints to refine the model.
3.  **Hybrid Modeling:** To improve performance without sacrificing interpretability, we train a powerful LightGBM model and then distill its knowledge into a simple, interpretable Decision Tree (a surrogate model).

## Dataset

This project uses the **"Diabetes 130-US hospitals for years 1999-2008 Data Set"** from the UCI Machine Learning Repository. It contains over 100,000 inpatient encounters.

- **Source:** [https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008](https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008)

## Repository Structure
human_centered_readmission/
├── README.md                 <- You are here
├── requirements.txt          <- Project dependencies
├── config.yml                <- Configuration file for paths and parameters
├── data/                     <- Raw and processed data
├── reports/                  <- Evaluation results, figures, and model rules
└── scripts/                  <- Python scripts for the entire pipeline

## Setup and Usage

**1. Clone the repository and create a virtual environment:**

```bash
git clone https://github.com/kelvin-jose/interpretableML-diabetes-risk.git

cd interpretableML-diabetes-risk

pip install -r requirements.txt

# Step 1: Download the data
python scripts/fetch_data.py

# Step 2: Preprocess the data
python scripts/preprocess.py

# Step 3: Train the initial interpretable model
python scripts/train_interpretable.py

# Step 4: Run the interactive session to refine the model
python scripts/interactive_session.py

# Step 5: Train the initial interpretable model but with constraints
python scripts/train_interpretable.py --refined

# Step 6: Train the hybrid model
python scripts/train_hybrid.py
```

