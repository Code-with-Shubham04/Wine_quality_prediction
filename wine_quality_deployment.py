import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("wine_quality_prediction.pkl")

st.set_page_config(page_title="Wine Quality Prediction", page_icon="🍷")

st.title("🍷 Wine Quality Prediction System")
st.write("Enter the chemical properties of the wine to predict its quality.")

st.subheader("Wine Chemical Properties")

# Input fields
fixed_acidity = st.number_input("Fixed Acidity", min_value=0.0, value=7.0)
volatile_acidity = st.number_input("Volatile Acidity", min_value=0.0, value=0.5)
citric_acid = st.number_input("Citric Acid", min_value=0.0, value=0.3)
residual_sugar = st.number_input("Residual Sugar", min_value=0.0, value=2.0)
chlorides = st.number_input("Chlorides", min_value=0.0, value=0.07)
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", min_value=0.0, value=15.0)
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", min_value=0.0, value=50.0)
density = st.number_input("Density", min_value=0.0, value=0.996)
pH = st.number_input("pH", min_value=0.0, value=3.3)
sulphates = st.number_input("Sulphates", min_value=0.0, value=0.6)
alcohol = st.number_input("Alcohol", min_value=0.0, value=10.0)

# Create dataframe (must match training columns exactly)
input_data = pd.DataFrame([[ 
    fixed_acidity,
    volatile_acidity,
    citric_acid,
    residual_sugar,
    chlorides,
    free_sulfur_dioxide,
    total_sulfur_dioxide,
    density,
    pH,
    sulphates,
    alcohol
]], columns=[
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol"
])

st.write("### Input Data")
st.dataframe(input_data)

# Prediction
if st.button("Predict Wine Quality 🍷"):

    prediction = model.predict(input_data)

    st.success(f"Predicted Wine Quality Score: {prediction[0]}")
