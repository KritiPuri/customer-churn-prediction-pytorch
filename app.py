import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
from model import ChurnANN

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊", layout="centered")
st.title("Customer Churn Prediction")
st.write("Enter customer details to estimate the probability that the customer will leave the bank.")

MODEL_PATH = "artifacts/churn_model.pth"
PREPROCESSOR_PATH = "artifacts/preprocessor.joblib"

@st.cache_resource
def load_assets():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(PREPROCESSOR_PATH):
        raise FileNotFoundError("Model files are missing. Run `python train.py` first.")
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    model = ChurnANN(
        checkpoint["input_dim"], checkpoint["hidden1"],
        checkpoint["hidden2"], checkpoint["dropout"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, preprocessor, checkpoint.get("threshold", 0.5)

try:
    model, preprocessor, threshold = load_assets()
except Exception as exc:
    st.error(str(exc))
    st.stop()

with st.form("customer_form"):
    geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
    gender = st.selectbox("Gender", ["Female", "Male"])
    c1, c2 = st.columns(2)
    with c1:
        credit_score = st.number_input("Credit score", min_value=300, max_value=900, value=650)
        age = st.number_input("Age", min_value=18, max_value=100, value=40)
        tenure = st.slider("Tenure (years)", 0, 10, 3)
        balance = st.number_input("Balance", min_value=0.0, value=60000.0, step=1000.0)
    with c2:
        num_products = st.selectbox("Number of products", [1, 2, 3, 4], index=1)
        has_card = st.selectbox("Has credit card?", ["Yes", "No"])
        active = st.selectbox("Is active member?", ["Yes", "No"])
        salary = st.number_input("Estimated salary", min_value=0.0, value=50000.0, step=1000.0)
    submitted = st.form_submit_button("Predict churn", type="primary")

if submitted:
    customer = pd.DataFrame([{
        "CreditScore": credit_score, "Geography": geography, "Gender": gender,
        "Age": age, "Tenure": tenure, "Balance": balance,
        "NumOfProducts": num_products, "HasCrCard": int(has_card == "Yes"),
        "IsActiveMember": int(active == "Yes"), "EstimatedSalary": salary,
    }])
    features = preprocessor.transform(customer).astype("float32")
    with torch.no_grad():
        logits = model(torch.tensor(features, dtype=torch.float32))
        probability = torch.sigmoid(logits).item()
    st.metric("Churn probability", f"{probability:.1%}")
    if probability >= threshold:
        st.warning("The model predicts that this customer is likely to churn.")
    else:
        st.success("The model predicts that this customer is unlikely to churn.")
    st.caption("This is a model estimate, not a guarantee. Use it as decision support.")
