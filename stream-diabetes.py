import pickle
import streamlit as st
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

# Muat model dan scaler
diabetes_model = pickle.load(open("diabetes_model.sav", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Title web
st.title("Data Mining Prediksi Diabetes")

# Membagi jadi 2 kolom
col1, col2 = st.columns(2)


with col1:
    Pregnancies = st.text_input("Input Nilai Pregnancies")
    Glucose = st.text_input("Input Nilai Glucose")
    BloodPressure = st.text_input("Input Nilai BloodPressure")
    SkinThickness = st.text_input("Input Nilai SkinThickness")
with col2:
    Insulin = st.text_input("Input Nilai Insulin")
    BMI = st.text_input("Input Nilai BMI")
    DiabetesPedigreeFunction = st.text_input("Input Nilai DiabetesPedigreeFunction")
    Age = st.text_input("Input Nilai Age")

# Prediksi
diab_diagnosis = ""


# Tombol submit
if st.button("Predict"):
    try:
        # Membuat DataFrame input
        input_features = pd.DataFrame([[float(Pregnancies), float(Glucose), float(BloodPressure), float(SkinThickness), 
                                        float(Insulin), float(BMI), float(DiabetesPedigreeFunction), float(Age)]],
                                      columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                                               'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])

        # Standardisasi data
        standardized_features = scaler.transform(input_features)

        # Prediksi
        diab_prediction = diabetes_model.predict(standardized_features)

        if diab_prediction[0] == 1:
            diab_diagnosis = "Diagnosed with Diabetes"
        else:
            diab_diagnosis = "Healthy"

        st.success(diab_diagnosis)

    except ValueError as e:
        st.error(f"Terjadi kesalahan dalam input data: {e}")
