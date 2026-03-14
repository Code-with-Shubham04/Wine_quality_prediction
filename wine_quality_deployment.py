import streamlit as st
import pandas as pd
import joblib

model = joblib.load("wine_quality_prediction.pkl")

st.title("Wine Quality Prediction System")

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

# Input list
values = [
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
]

# Use model feature names
df = pd.DataFrame([values], columns=model.feature_names_in_)

if st.button("Predict Wine Quality"):
    prediction = model.predict(df)
    st.success(f"Predicted Wine Quality: {prediction[0]}")
