import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

DROP_COLUMNS = ["RowNumber", "CustomerId", "Surname"]
NUMERIC_COLUMNS = [
    "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
    "HasCrCard", "IsActiveMember", "EstimatedSalary"
]

def build_preprocessor():
    # Keep geography as one-hot encoded and standardize numeric features.
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_COLUMNS),
            ("geography", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["Geography"]),
            ("gender", OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False), ["Gender"]),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

def prepare_features(frame: pd.DataFrame):
    return frame.drop(columns=DROP_COLUMNS + ["Exited"], errors="ignore")
