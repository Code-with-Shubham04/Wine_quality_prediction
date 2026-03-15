import streamlit as st
import numpy as np
import joblib

# Load trained model
model = joblib.load("wine_quality_prediction.pkl")

st.title("🍷 Wine Quality Prediction System")

st.write("Enter the chemical properties of wine")

# Inputs
fixed_acidity = st.number_input("Fixed Acidity", 0.0)
volatile_acidity = st.number_input("Volatile Acidity", 0.0)
citric_acid = st.number_input("Citric Acid", 0.0)
residual_sugar = st.number_input("Residual Sugar", 0.0)
chlorides = st.number_input("Chlorides", 0.0)
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", 0.0)
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", 0.0)
density = st.number_input("Density", 0.0)
pH = st.number_input("pH", 0.0)
sulphates = st.number_input("Sulphates", 0.0)
alcohol = st.number_input("Alcohol", 0.0)

# Wine type
wine_type = st.selectbox("Wine Type", ["Red", "White"])

# Encode type
type_encoded = 0 if wine_type == "Red" else 1

# Prediction
if st.button("Predict Wine Quality"):

    input_data = np.array([[ 
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
        alcohol,
        type_encoded
    ]])

    prediction = model.predict(input_data)

    st.success(f"Predicted Wine Quality: {prediction[0]}")

    if prediction[0] >= 7:
        st.success("🍷 Good Quality Wine")
    elif prediction[0] >= 5:
        st.warning("🍷 Average Quality Wine")
    else:
        st.error("🍷 Low Quality Wine")
