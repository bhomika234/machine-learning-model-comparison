import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="ML Prediction App",
    page_icon="🤖",
    layout="centered"
)

# Title
st.title("🤖 Machine Learning Prediction App")
st.write("Random Forest based prediction system")

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("best_model.pkl")

model = load_model()

st.success("Model loaded successfully!")

st.subheader("Enter Input Data")

# -------------------------------------------------
# IMPORTANT:
# Replace these input fields with your actual
# dataset features.
# -------------------------------------------------

feature_1 = st.number_input("Feature 1", value=0.0)
feature_2 = st.number_input("Feature 2", value=0.0)
feature_3 = st.number_input("Feature 3", value=0.0)

# Prediction button
if st.button("Predict"):

    input_data = pd.DataFrame({
        "Feature_1": [feature_1],
        "Feature_2": [feature_2],
        "Feature_3": [feature_3]
    })

    prediction = model.predict(input_data)

    st.subheader("Prediction Result")

    st.success(f"Prediction: {prediction[0]}")