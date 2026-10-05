import os
import json
import joblib
import pandas as pd

from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

try:
    from src.preprocess import preprocess_data

except ModuleNotFoundError:
    from preprocess import preprocess_data
 
os.makedirs("models", exist_ok=True)

print("\nLoading Dataset...")

X, y, feature_names = preprocess_data("data/alzheimers_disease_data.csv")

if not os.path.exists("models/top_features.json"):
    raise FileNotFoundError("Run feature_selection.py first")

with open("models/top_features.json", "r") as f:
    selected_features = json.load(f)

X = X[selected_features]
feature_ranges = {}

for feature in selected_features:

    feature_ranges[feature] = {
        "min": round(float(X[feature].min())),
        "max": round(float(X[feature].max())),
    }

with open("models/feature_ranges.json", "w") as f:

    json.dump(feature_ranges, f, indent=4)

print("\nFeature Ranges Saved")

print(f"\nSelected Features: {len(selected_features)}")

print("\nApplying MinMax Scaling...")

scaler = MinMaxScaler()

X = scaler.fit_transform(X)

joblib.dump(scaler, "models/scaler.pkl")

print("Scaler Saved")

print("\nSplitting Dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nApplying SMOTE...")

smote = SMOTE(random_state=42)

X_train, y_train = smote.fit_resample(X_train, y_train)

print(f"Training Samples After SMOTE: {len(X_train)}")

print("\nTraining Final XGBoost Model...")

model = XGBClassifier(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
)

model.fit(X_train, y_train)

print("Training Completed")

joblib.dump(model, "models/xgboost_model.pkl")

print("Model Saved")

print("\nEvaluating Model...")

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

precision = precision_score(y_test, predictions)

recall = recall_score(y_test, predictions)

f1 = f1_score(y_test, predictions)

print("\nModel Performance")

print(f"Accuracy : {accuracy*100:.2f}%")

print(f"Precision: {precision*100:.2f}%")

print(f"Recall   : {recall*100:.2f}%")

print(f"F1 Score : {f1*100:.2f}%")

print("\nClassification Report\n")

print(classification_report(y_test, predictions))

metrics = {
    "model": "Boruta + XGBoost",
    "accuracy": round(accuracy * 100, 2),
    "precision": round(precision * 100, 2),
    "recall": round(recall * 100, 2),
    "f1_score": round(f1 * 100, 2),
}

with open("models/model_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("\nMetrics Saved")

feature_importance = {}

for feature, importance in zip(selected_features, model.feature_importances_):
    feature_importance[feature] = float(importance)

with open("models/feature_importance.json", "w") as f:
    json.dump(feature_importance, f, indent=4)

metadata = {
    "model": "Boruta + XGBoost",
    "num_features": len(selected_features),
    "features": selected_features,
    "feature_importance": feature_importance,
}

with open("models/model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

print("\nMetadata Saved")

print("\nFeature Importance Saved")

print("\nTraining Completed Successfully")
