import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model
model = joblib.load("models/student_model.pkl")

# App title
st.set_page_config(page_title="🎓 Student Performance Predictor", layout="centered")
st.title("🎓 Student Performance Prediction")
st.write("Predict whether a student will **Pass or Fail** based on given inputs.")

# Sidebar for input
st.sidebar.header("📌 Enter Student Details")

# Example inputs (change based on dataset features)
age = st.sidebar.slider("Age", 15, 22, 18)
studytime = st.sidebar.selectbox("Weekly Study Time (hrs)", [1, 2, 3, 4])
failures = st.sidebar.slider("Past Failures", 0, 5, 0)
absences = st.sidebar.slider("Absences", 0, 30, 2)
health = st.sidebar.slider("Health (1=Bad, 5=Excellent)", 1, 5, 3)

# Convert inputs into dataframe
input_data = pd.DataFrame({
    "age": [age],
    "studytime": [studytime],
    "failures": [failures],
    "absences": [absences],
    "health": [health]
})

# Prediction
if st.button("🔮 Predict Performance"):
    prediction = model.predict(input_data)[0]
    result = "✅ Pass" if prediction == 1 else "❌ Fail"
    st.subheader(f"Prediction: {result}")

    
# python -m streamlit run app.py
