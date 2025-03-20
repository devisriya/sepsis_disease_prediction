import streamlit as st
import pandas as pd
import catboost as cb
import numpy as np

# Load the trained CatBoost model
MODEL_PATH = "catboost_sepsis_model.cbm"

@st.cache_resource
def load_model():
    model = cb.CatBoostClassifier()
    model.load_model(MODEL_PATH)
    return model

model = load_model()

# Streamlit UI
st.title("🔬 Sepsis Prediction App")
st.write("Enter patient details to predict the likelihood of sepsis.")

# Input fields based on dataset columns
feature_names = ["HR", "Resp", "Temp", "DBP", "SBP", "MAP", "O2Sat", "Age", "Gender"]

input_data = []
for feature in feature_names:
    value = st.number_input(f"{feature}", value=0.0, format="%.2f")
    input_data.append(value)

# Prediction button
if st.button("🔍 Predict Sepsis"):
    input_array = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_array)
    prediction_label = "Sepsis Detected ⚠" if prediction[0] == 1 else "No Sepsis ✅"
    st.subheader(prediction_label)