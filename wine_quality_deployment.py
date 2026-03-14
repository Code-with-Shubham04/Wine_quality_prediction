import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("wine_quality_prediction.pkl")

st.title("🍷 Wine Quality Prediction System")

st.write("Enter wine chemical properties")

# Inputs
fixed_acidity = st.number_input("Fixed Acidity", min_value=0.0)
volatile_acidity = st.number_input("Volatile Acidity", min_value=0.0)
citric_acid = st.number_input("Citric Acid", min_value=0.0)
residual_sugar = st.number_input("Residual Sugar", min_value=0.0)
chlorides = st.number_input("Chlorides", min_value=0.0)
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", min_value=0.0)
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", min_value=0.0)
density = st.number_input("Density", min_value=0.0)
pH = st.number_input("pH", min_value=0.0)
sulphates = st.number_input("Sulphates", min_value=0.0)
alcohol = st.number_input("Alcohol", min_value=0.0)

# Wine type
wine_type = st.selectbox("Wine Type", ["red", "white"])

# Encode type (same as training)
type_encoded = 0 if wine_type == "red" else 1

# Create dataframe with EXACT column order used in training
df = pd.DataFrame([[ 
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
    "alcohol",
    "type"
])

# Prediction
if st.button("Predict Wine Quality"):
    prediction = model.predict(df)
    st.success(f"Predicted Wine Quality: {prediction[0]}")
