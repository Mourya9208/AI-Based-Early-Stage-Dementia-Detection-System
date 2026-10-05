import os
import json

from boruta import BorutaPy

from sklearn.ensemble import RandomForestClassifier

try:

    from src.preprocess import preprocess_data

except ModuleNotFoundError:

    from preprocess import preprocess_data


os.makedirs("models", exist_ok=True)

print("\nStarting Boruta Feature Selection...")

X, y, feature_names = preprocess_data("data/alzheimers_disease_data.csv")

print(f"Dataset Loaded: {len(X)} samples")

print("\nTraining Random Forest...")

rf = RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=42, n_jobs=-1)

boruta = BorutaPy(estimator=rf, n_estimators="auto", verbose=2, random_state=42)

print("\nRunning Boruta...")

boruta.fit(X, y.values)

selected_features = []

for feature, selected in zip(feature_names, boruta.support_):

    if selected:

        selected_features.append(feature)

ranking = {}

for feature, rank in zip(feature_names, boruta.ranking_):

    ranking[feature] = int(rank)

print("\nSelected Features:\n")

for idx, feature in enumerate(selected_features, start=1):

    print(f"{idx}. {feature}")

if len(selected_features) > 10:

    selected_features = selected_features[:10]

with open("models/top_features.json", "w") as f:

    json.dump(selected_features, f, indent=4)

with open("models/feature_ranking.json", "w") as f:

    json.dump(ranking, f, indent=4)

print("\nSaved:")

print("models/top_features.json")

print("models/feature_ranking.json")

print("\nBoruta Feature Selection Completed Successfully")
