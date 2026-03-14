# Wine Quality Prediction Streamlit App

import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("wine_quality_prediction.pkl")

# App Title
st.title("🍷 Wine Quality Prediction System")

st.write("Enter the chemical properties of wine to predict its quality.")

# User Inputs
fixed_acidity = st.number_input("Fixed Acidity")
volatile_acidity = st.number_input("Volatile Acidity")
citric_acid = st.number_input("Citric Acid")
residual_sugar = st.number_input("Residual Sugar")
chlorides = st.number_input("Chlorides")
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide")
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide")
density = st.number_input("Density")
pH = st.number_input("pH")
sulphates = st.number_input("Sulphates")
alcohol = st.number_input("Alcohol")

# Create DataFrame with correct column names
input_data = pd.DataFrame({
    "fixed_acidity": [fixed_acidity],
    "volatile_acidity": [volatile_acidity],
    "citric_acid": [citric_acid],
    "residual_sugar": [residual_sugar],
    "chlorides": [chlorides],
    "free_sulfur_dioxide": [free_sulfur_dioxide],
    "total_sulfur_dioxide": [total_sulfur_dioxide],
    "density": [density],
    "pH": [pH],
    "sulphates": [sulphates],
    "alcohol": [alcohol]
})

# Prediction Button
if st.button("Predict Wine Quality"):
    prediction = model.predict(input_data)

    st.success(f"Predicted Wine Quality: {prediction[0]}")
