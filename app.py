"""
Bonus: Simple Streamlit prediction interface for the Credit Card Fraud
Detection model saved in the notebook (best_model.pkl).

Run with:
    streamlit run app.py

Make sure best_model.pkl is in the same folder as this script
(it is produced by the "Best Model Save" section of the notebook).
"""

import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Fraud Detection", page_icon="💳")

st.title("💳 Credit Card Fraud Prediction")
st.write(
    "Enter transaction details below and the trained Random Forest model "
    "will predict whether the transaction is **Normal** or **Fraudulent**."
)

# ---- Load the trained model ----
@st.cache_resource
def load_model():
    return joblib.load("best_model.pkl")

try:
    model = load_model()
except FileNotFoundError:
    st.error(
        "best_model.pkl not found. Run the notebook's 'Best Model Save' "
        "section first so this file is created in the same folder."
    )
    st.stop()

# ---- Build inputs dynamically from the features the model was trained on ----
st.subheader("Transaction Details")

feature_names = list(model.feature_names_in_)
input_values = {}

# Two columns so the form isn't too long
col1, col2 = st.columns(2)
for i, feature in enumerate(feature_names):
    target_col = col1 if i % 2 == 0 else col2
    input_values[feature] = target_col.number_input(feature, value=0.0, format="%.4f")

# ---- Predict ----
if st.button("Predict"):
    input_df = pd.DataFrame([input_values])[feature_names]
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1] if hasattr(model, "predict_proba") else None

    if prediction == 1:
        st.error(f"⚠️ Prediction: **Fraudulent Transaction**")
    else:
        st.success(f"✅ Prediction: **Normal Transaction**")

    if proba is not None:
        st.write(f"Fraud probability: **{proba:.2%}**")
