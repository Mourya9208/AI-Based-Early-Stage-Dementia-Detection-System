import json
import joblib
import pandas as pd

# =========================
# Load Model Files
# =========================

xgb_model = joblib.load("models/xgboost_model.pkl")

scaler = joblib.load("models/scaler.pkl")

with open("models/model_metadata.json", "r") as f:

    metadata = json.load(f)

FEATURES = metadata["features"]


# =========================
# Prediction Explanation
# =========================


def generate_explanation(probability, feature_dict):

    explanation = ""

    if probability >= 0.75:

        explanation += (
            "Patient shows strong indicators " "associated with dementia.\n\n"
        )

    elif probability >= 0.50:

        explanation += "Patient shows moderate dementia " "risk indicators.\n\n"

    else:

        explanation += "Patient shows relatively low " "dementia risk.\n\n"

    # MMSE

    mmse = float(feature_dict["mmse"])

    if mmse < 10:

        explanation += (
            f"- MMSE score ({mmse}) is very low, "
            f"indicating severe cognitive impairment.\n"
        )

    elif mmse < 20:

        explanation += (
            f"- MMSE score ({mmse}) suggests " f"moderate cognitive decline.\n"
        )

    elif mmse < 24:

        explanation += (
            f"- MMSE score ({mmse}) indicates " f"mild cognitive impairment.\n"
        )

    else:

        explanation += (
            f"- MMSE score ({mmse}) is within " f"the normal cognitive range.\n"
        )

    # Functional Assessment

    fa = float(feature_dict["functionalassessment"])

    if fa < 3:

        explanation += (
            f"- Functional Assessment score ({fa}) "
            f"indicates significant difficulty "
            f"in daily functioning.\n"
        )

    elif fa < 5:

        explanation += (
            f"- Functional Assessment score ({fa}) "
            f"suggests reduced functional ability.\n"
        )

    else:

        explanation += (
            f"- Functional Assessment score ({fa}) "
            f"shows relatively preserved functionality.\n"
        )

    # Memory Complaints

    mc = float(feature_dict["memorycomplaints"])

    if mc == 1:

        explanation += (
            "- Memory complaints are reported, "
            "which is a common dementia indicator.\n"
        )

    else:

        explanation += "- No memory complaints were reported.\n"

    # Behavioral Problems

    bp = float(feature_dict["behavioralproblems"])

    if bp == 1:

        explanation += (
            "- Behavioral problems are present, "
            "which may be associated with dementia.\n"
        )

    else:

        explanation += "- No behavioral problems were reported.\n"

    # ADL

    adl = float(feature_dict["adl"])

    if adl < 3:

        explanation += (
            f"- ADL score ({adl}) indicates severe "
            f"difficulty performing daily activities.\n"
        )

    elif adl < 5:

        explanation += (
            f"- ADL score ({adl}) suggests reduced "
            f"independence in daily activities.\n"
        )

    else:

        explanation += (
            f"- ADL score ({adl}) indicates good "
            f"ability to perform daily activities.\n"
        )

    return explanation


# =========================
# Prediction Function
# =========================


def predict(data_dict):

    input_df = pd.DataFrame([data_dict])

    input_df = input_df[FEATURES]

    scaled_data = scaler.transform(input_df)

    probability = float(xgb_model.predict_proba(scaled_data)[0][1])

    confidence = round(probability * 100, 2)

    if probability >= 0.75:

        prediction = "Demented"

        risk_level = "VERY HIGH"

    elif probability >= 0.50:

        prediction = "High Risk"

        risk_level = "HIGH"

    elif probability >= 0.30:

        prediction = "Moderate Risk"

        risk_level = "MODERATE"

    else:

        prediction = "Normal"

        risk_level = "LOW"

    explanation = generate_explanation(probability, data_dict)

    return {
        "prediction": prediction,
        "risk_level": risk_level,
        "risk_score": round(probability, 4),
        "confidence": confidence,
        "explanation": explanation,
    }
