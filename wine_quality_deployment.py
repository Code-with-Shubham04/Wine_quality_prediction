import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("wine_quality_prediction.pkl")

st.title("🍷 Wine Quality Prediction System")
st.write("Enter wine chemical properties")

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

wine_type = st.selectbox("Wine Type", ["red", "white"])
type_encoded = 0 if wine_type == "red" else 1

# Create dataframe
data = {
    "fixed acidity": fixed_acidity,
    "volatile acidity": volatile_acidity,
    "citric acid": citric_acid,
    "residual sugar": residual_sugar,
    "chlorides": chlorides,
    "free sulfur dioxide": free_sulfur_dioxide,
    "total sulfur dioxide": total_sulfur_dioxide,
    "density": density,
    "pH": pH,
    "sulphates": sulphates,
    "alcohol": alcohol,
    "type": type_encoded
}

df = pd.DataFrame([data])

# Reorder columns exactly as model expects
df = df[model.feature_names_in_]

if st.button("Predict Wine Quality"):
    prediction = model.predict(df)
    st.success(f"Predicted Wine Quality: {prediction[0]}")
