from flask import Flask, render_template, request

import json

from src.predict import predict

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def home():

    with open("models/top_features.json", "r") as f:

        features = json.load(f)

    with open("models/model_metrics.json", "r") as f:

        metrics = json.load(f)

    with open("models/feature_ranges.json", "r") as f:

        feature_ranges = json.load(f)

    result = None

    if request.method == "POST":

        try:

            patient_data = {}

            for feature in features:

                value = float(request.form[feature])

                if feature == "mmse":

                    if value < 0 or value > 30:

                        raise ValueError("MMSE must be between 0 and 30")

                elif feature in feature_ranges:

                    min_val = feature_ranges[feature]["min"]

                    max_val = feature_ranges[feature]["max"]

                    if value < min_val or value > max_val:

                        raise ValueError(
                            f"{feature} must be between "
                            f"{round(min_val,2)} and "
                            f"{round(max_val,2)}"
                        )

                patient_data[feature] = value

            result = predict(patient_data)

        except Exception as e:

            result = {
                "prediction": "Error",
                "risk_level": "N/A",
                "risk_score": 0,
                "confidence": 0,
                "important_features": [],
                "explanation": str(e),
            }

    return render_template(
        "index.html",
        features=features,
        metrics=metrics,
        feature_ranges=feature_ranges,
        result=result,
    )


if __name__ == "__main__":

    app.run(debug=True, host="0.0.0.0", port=5000)
