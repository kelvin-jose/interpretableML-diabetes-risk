# Human-Centered Interpretable Modeling for Diabetic Readmission Risk

This project explores the development of interpretable machine learning models for predicting 30-day hospital readmission for diabetic patients. It is designed to demonstrate an understanding of human-centered AI, pattern-based modeling, and hybrid approaches that balance interpretability and predictive performance, aligning with research in Human-Centered Interpretable ML.

## Project Overview

The core idea is to move beyond black-box models and create a workflow where a domain expert can interact with, guide, and trust the model's predictions.

1.  **Interpretable Baseline:** We first train a rule-based model (`skope-rules`) that generates simple, logical IF-THEN rules.
2.  **Human-in-the-Loop:** An interactive script allows a user to review these rules, provide feedback on their clinical relevance, and generate constraints to refine the model.
3.  **Hybrid Modeling:** To improve performance without sacrificing interpretability, we train a powerful LightGBM model and then distill its knowledge into a simple, interpretable Decision Tree (a surrogate model).


