# Titanic Survival Prediction: Machine Learning Classification #

An end-to-end machine learning classification project using Python and scikit-learn to predict passenger survival from the Titanic dataset.

## Project Overview

This project demonstrates a complete machine learning workflow, covering exploratory data analysis, feature selection, preprocessing, model development, cross-validation, hyperparameter tuning, model comparison, and final evaluation on an unseen test set.

The project was developed in Python using pandas and scikit-learn.

## Objective

The objective is to predict whether a Titanic passenger survived the disaster.

Target variable: survived
- 0 = Did not survive
- 1 = Survived
## Dataset

The dataset contains 891 observations and 15 variables, including passenger class, sex, age, family-related variables, fare, and embarkation port.

The predictors used for modelling were **pclass, sex, age, sibsp, parch, fare, and embarked.**

Variables that could cause data leakage, were redundant, derived from other variables, or had substantial missingness were excluded.

# Machine Learning Workflow

The project follows the workflow:

**Data Understanding → Exploratory Data Analysis → Feature Selection → Train-Test Split → Preprocessing → Model Development → Cross-Validation → Hyperparameter Tuning → Model Comparison → Final Model Selection → Held-Out Test Evaluation**

## Preprocessing

The modelling pipeline includes median imputation for numerical variables, most-frequent imputation for categorical variables, standardization of numerical variables, and one-hot encoding of categorical variables.

All preprocessing steps were integrated into scikit-learn pipelines so that they were learned only from the training data during model development and cross-validation.

## Models Evaluated

Five classification algorithms were evaluated: Logistic Regression, Decision Tree, Random Forest, K-Nearest Neighbors (KNN), and Gradient Boosting.

We conducted hyperparameter experiments for the Decision Tree, Random Forest, KNN, and Gradient Boosting models.

## Cross-Validation Results

The candidate models showed broadly similar cross-validation performance.

**Logistic Regression** achieved 79.63% accuracy, 75.00% precision, 70.33% recall, 72.59% F1-score, and a ROC-AUC of 0.851.

**The tuned Decision Tree** achieved 81.89% accuracy, 81.03% precision, 68.86% recall, 74.46% F1-score, and a ROC-AUC of 0.801.

**The tuned Random Forest** achieved 81.88% accuracy, 81.58% precision, 68.13% recall, 74.25% F1-score, and a ROC-AUC of 0.867.

**The tuned KNN** achieved 81.74% accuracy, 80.17% precision, 69.60% recall, 74.51% F1-score, and a ROC-AUC of 0.865.

**The baseline Gradient Boosting** model achieved 81.60% accuracy, 81.42% precision, 67.40% recall, 73.75% F1-score, and the highest ROC-AUC of 0.873.

**The tuned Gradient Boosting** model achieved 81.88% accuracy, 84.62% precision, 64.47% recall, 73.18% F1-score, and a ROC-AUC of 0.866.

Based primarily on cross-validated ROC-AUC and its overall performance profile, the baseline **Gradient Boosting model was selected as the final candidate**.

## Final Test Evaluation

The selected Gradient Boosting model was then evaluated once on an untouched test set containing 20% of the observations.

On the held-out test set, the model achieved 79.89% accuracy, 78.95% precision, 65.22% recall, 71.43% F1-score, and a ROC-AUC of 0.818.

The difference between cross-validation and test performance reflects expected variation when evaluating a model on previously unseen data. The test set was not used for model selection or hyperparameter tuning.

## Repository Contents

The repository currently contains the polished Jupyter notebook with the complete analysis and modelling workflow.

Additional supporting files, including figures, model results, and dependency information, can be added as the project is further organized.

## Technologies

The project uses Python, pandas, NumPy, scikit-learn, Matplotlib, Seaborn, Google Colab, and GitHub.

## Future Work

Future work will extend the project toward model interpretability and explainable machine learning, including SHAP-based analysis.
