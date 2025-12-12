import streamlit as st
import joblib
import numpy as np
import os

st.set_page_config(page_title="Boston House Price Predictor", layout="centered")

st.title("🏠 Boston House Price Predictor")

model_path = "model/model.pkl"

# Load the trained model
if not os.path.exists(model_path):
    st.error("Model not found! Run train_model.py first to generate model/model.pkl")
else:
    model = joblib.load(model_path)

    st.subheader("Enter House Features")

    # Minimal feature set (mapped to your model's features)
    rm = st.number_input("Average Number of Rooms (RM)", min_value=1.0, max_value=10.0, value=6.0, step=0.1)
    age = st.number_input("Age of the Property (AGE)", min_value=1.0, max_value=100.0, value=40.0, step=1.0)
    tax = st.number_input("Property Tax Rate (TAX)", min_value=100.0, max_value=800.0, value=300.0, step=10.0)
    lstat = st.number_input("% Lower Status Population (LSTAT)", min_value=1.0, max_value=40.0, value=12.0, step=0.5)

    if st.button("Predict Price"):
        # Arrange input into correct shape
        features = np.array([[rm, age, tax, lstat]])

        # Predict using model
        prediction = model.predict(features)[0]

        # Boston MEDV is in thousands of dollars
        st.success(f"🏡 Estimated Price: **${prediction * 1000:,.2f}**")
