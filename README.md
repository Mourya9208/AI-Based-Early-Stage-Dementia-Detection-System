# AI-Based Early-Stage Dementia Detection System

## Overview

The AI-Based Early-Stage Dementia Detection System is a machine learning-based web application designed to predict dementia risk from clinical and cognitive assessment data.

The system uses an XGBoost classification model along with feature selection, MinMax scaling, and SMOTE-based class balancing. A Flask web application provides a user-friendly interface for entering patient assessment values and receiving real-time dementia risk predictions.

## Objectives

- Develop an AI-based system for early-stage dementia risk prediction.
- Analyze relevant clinical and cognitive assessment parameters.
- Apply feature selection to identify important input features.
- Handle class imbalance using SMOTE.
- Train and evaluate an XGBoost classification model.
- Provide real-time predictions through a Flask web application.
- Display prediction probability and risk information to support early screening.

## Technologies Used

- Python
- XGBoost
- Scikit-learn
- Pandas
- NumPy
- Flask
- imbalanced-learn
- Joblib
- HTML
- CSS

## Machine Learning Workflow

```text
Clinical Dataset
       |
       v
Data Preprocessing
       |
       v
Feature Selection
       |
       v
MinMax Scaling
       |
       v
Train-Test Split
       |
       v
SMOTE Class Balancing
       |
       v
XGBoost Model Training
       |
       v
Model Evaluation
       |
       v
Flask Web Application
       |
       v
Dementia Risk Prediction
Machine Learning Model
The project uses XGBoost as the primary classification algorithm for predicting dementia status.
Class Balancing
SMOTE (Synthetic Minority Over-sampling Technique) is applied to the training data to address class imbalance by generating synthetic samples for the minority class.
Feature Scaling
MinMaxScaler is used to normalize the selected input features before model training and prediction.
Model Performance
The trained model achieved the following performance on the test dataset:
Metric	Score
Accuracy	94.19%
F1-Score	91.80%


Dataset size: 2,150 clinical records.
Web Application
The Flask-based web application allows users to enter the required patient assessment values.
The application processes the input through the trained scaler and XGBoost model and provides a real-time dementia risk prediction.

Project Structure
AI-Based-Early-Stage-Dementia-Detection-System/
│
├── data/
│   └── alzheimers_disease_data.csv
│
├── models/
│   ├── feature_importance.json
│   ├── feature_ranges.json
│   ├── feature_ranking.json
│   ├── feature_selection_metadata.json
│   ├── model_metadata.json
│   ├── model_metrics.json
│   ├── scaler.pkl
│   ├── top_features.json
│   └── xgboost_model.pkl
│
├── src/
│   ├── __init__.py
│   ├── feature_selection.py
│   ├── predict.py
│   ├── preprocess.py
│   └── train_xgboost.py
│
├── static/
│   └── css/
│       └── style.css
│
├── templates/
│   └── index.html
│
├── app.py
├── README.md
└── .gitignore