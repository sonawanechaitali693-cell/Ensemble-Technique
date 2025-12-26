**Breast Cancer Classification using Decision Tree & XGBoost**

This project demonstrates binary classification on the Breast Cancer Wisconsin dataset using:

A Single Decision Tree Classifier

An Ensemble Learning model (XGBoost)

The goal is to compare the performance of a simple tree-based model with a powerful boosting-based ensemble model.

Project Overview:

Dataset: Breast Cancer Wisconsin (Diagnostic)

Problem Type: Binary Classification

0 → Malignant

1 → Benign

Models Used:

Decision Tree Classifier

XGBoost Classifier

Evaluation Metrics:

Accuracy

Precision, Recall, F1-score

Feature Importance (XGBoost)

 Dataset Details:

The dataset is loaded directly from scikit-learn:

Total Samples: 569

Features: 30 numeric features

Target Classes: 2 (Malignant, Benign)

Example features:

mean radius

mean texture

mean perimeter

mean area

worst concavity

Tech Stack & Libraries:

Python
pandas
scikit-learn
xgboost


Install XGBoost (if not already installed):

pip install xgboost

 Model Pipeline
 1️. Data Loading & Splitting

Dataset loaded using load_breast_cancer()

Train-Test split:

70% Training

30% Testing

Random state fixed for reproducibility

2️. Decision Tree Classifier
DecisionTreeClassifier(max_depth=4, random_state=42)


A simple tree-based model

Limited depth to prevent overfitting

Used as a baseline model

3️. XGBoost Classifier (Ensemble Learning)
XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    eval_metric='logloss',
    random_state=42
)


Uses Gradient Boosting

Combines multiple weak learners

Optimized for higher accuracy and generalization

 Model Comparison Results:
 
Single Decision Tree Accuracy: 0.92xx
XGBoost Ensemble Accuracy:     0.96xx


 XGBoost outperforms the single Decision Tree, showcasing the power of ensemble learning.

 XGBoost Classification Report

The classification report includes:

Precision

Recall

F1-score

Support

For both classes:

Malignant

Benign

This gives a deeper insight beyond accuracy.

Feature Importance (XGBoost):

One major advantage of tree-based models is interpretability.

Top predictive features typically include:

worst radius

mean concave points

worst perimeter

mean area

worst concavity

These features contribute most to model decisions.

 Key Learnings:

Decision Trees are easy to understand but can underperform alone

XGBoost significantly improves accuracy via boosting

Ensemble methods reduce bias and variance

Feature importance helps in medical interpretability

Future Improvements:

Cross-validation

Hyperparameter tuning (GridSearch / RandomSearch)

ROC-AUC evaluation

Model explainability using SHAP

Deployment using Flask / FastAPI

 Conclusion:

This project clearly demonstrates how ensemble learning (XGBoost) provides superior performance compared to a single Decision Tree, especially in critical domains like medical diagnosis.
