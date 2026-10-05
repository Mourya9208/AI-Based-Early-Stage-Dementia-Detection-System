import pandas as pd


def preprocess_data(path):

    df = pd.read_csv(path)

    df.columns = df.columns.str.strip().str.lower()

    if "diagnosis" not in df.columns:

        raise KeyError("Diagnosis column not found")

    y = df["diagnosis"]

    drop_columns = ["diagnosis", "patientid", "doctorincharge"]

    X = df.drop(columns=drop_columns, errors="ignore")

    binary_columns = []

    for col in X.columns:

        unique_values = set(X[col].dropna().unique())

        if unique_values.issubset({0, 1}):

            binary_columns.append(col)

    numeric_columns = [col for col in X.columns if col not in binary_columns]

    for col in binary_columns:

        X[col] = X[col].fillna(X[col].mode()[0])

    for col in numeric_columns:

        X[col] = X[col].fillna(X[col].mean())

    return (X, y, X.columns.tolist())
