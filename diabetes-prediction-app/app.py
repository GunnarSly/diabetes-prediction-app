import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import joblib
from tensorflow.keras.models import load_model

# ---------------------------
# Load all models automatically
# ---------------------------
def load_all_models():
    models = {}
    for file in os.listdir("models"):
        path = os.path.join("models", file)

        if file.endswith(".pkl"):
            try:
                models[file] = pickle.load(open(path, "rb"))
            except:
                pass

        elif file.endswith(".joblib"):
            try:
                models[file] = joblib.load(path)
            except:
                pass

        elif file.endswith(".h5"):
            try:
                models[file] = load_model(path)
            except:
                pass

    return models

models = load_all_models()


# ---------------------------
# UI Setup
# ---------------------------
st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="centered"
)

st.title("Diabetes Prediction System")
st.markdown("""
Enter the patient's laboratory values below to predict diabetes  
using one of the available trained machine learning models.
""")

st.divider()


# ---------------------------
# Sidebar: Model Selection
# ---------------------------
st.sidebar.header("Select a Model")
selected_model = st.sidebar.selectbox(
    "Available Models:",
    list(models.keys())
)

st.sidebar.success(f"Model selected: {selected_model}")


# ---------------------------
# Patient Data Input (No defaults)
# ---------------------------
st.subheader("Patient Input Data")

col1, col2 = st.columns(2)

with col1:
    preg = st.number_input("Pregnancies", min_value=0, step=1)
    glucose = st.number_input("Glucose Level", min_value=0)
    bp = st.number_input("Blood Pressure", min_value=0)
    skin = st.number_input("Skin Thickness", min_value=0)

with col2:
    insulin = st.number_input("Insulin Level", min_value=0)
    bmi = st.number_input("BMI", min_value=0.0)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, format="%.4f")
    age = st.number_input("Age", min_value=0, step=1)

st.divider()


# ---------------------------
# Prediction Button
# ---------------------------
if st.button("Predict", use_container_width=True):

    input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    model = models[selected_model]

    # Handle model differences
    try:
        prediction = model.predict(input_data)
    except:
        prediction = model.predict(input_data)[0][0]

    # Ensure numeric
    prediction_value = float(prediction)

    result = "Diabetic" if prediction_value > 0.5 else "Not Diabetic"

    st.success(f"Prediction Result: {result}")
    st.info(f"Model Raw Output: {prediction_value}")
