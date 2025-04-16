import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.title('Loan Approval Prediction App')

# Load models
model_options = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "Random Forest": "random_forest.pkl"
}

selected_model_name = st.sidebar.selectbox("Select Model", list(model_options.keys()))
model_path = model_options[selected_model_name]

@st.cache_resource
def load_model(path):
    with open(path, 'rb') as file:
        return pickle.load(file)

model = load_model(model_path)

# Load expected feature names used during training
try:
    with open("train_features.pkl", "rb") as f:
        expected_features = pickle.load(f)
except FileNotFoundError:
    st.error("Required file 'train_features.pkl' not found. Please ensure it is uploaded.")
    st.stop()

# Input fields
st.sidebar.header('Enter Loan Applicant Details')

Gender = st.sidebar.selectbox("Gender", ['Male', 'Female'])
Married = st.sidebar.selectbox("Married", ['Yes', 'No'])
Dependents = st.sidebar.selectbox("Dependents", ['0', '1', '2', '3+'])
Education = st.sidebar.selectbox("Education", ['Graduate', 'Not Graduate'])
Self_Employed = st.sidebar.selectbox("Self Employed", ['Yes', 'No'])
ApplicantIncome = st.sidebar.number_input("Applicant Income", min_value=0)
CoapplicantIncome = st.sidebar.number_input("Coapplicant Income", min_value=0)
LoanAmount = st.sidebar.number_input("Loan Amount", min_value=0)
Loan_Amount_Term = st.sidebar.number_input("Loan Amount Term", min_value=12, step=12)
Credit_History = st.sidebar.selectbox("Credit History", [1.0, 0.0])
Property_Area = st.sidebar.selectbox("Property Area", ['Urban', 'Rural', 'Semiurban'])

# Prepare input data for prediction
input_dict = {
    'Gender': Gender,
    'Married': Married,
    'Dependents': Dependents,
    'Education': Education,
    'Self_Employed': Self_Employed,
    'ApplicantIncome': ApplicantIncome,
    'CoapplicantIncome': CoapplicantIncome,
    'LoanAmount': LoanAmount,
    'Loan_Amount_Term': Loan_Amount_Term,
    'Credit_History': Credit_History,
    'Property_Area': Property_Area
}

input_df = pd.DataFrame([input_dict])
input_encoded = pd.get_dummies(input_df, drop_first=True)

# Add missing columns and align order
for col in expected_features:
    if col not in input_encoded.columns:
        input_encoded[col] = 0
input_encoded = input_encoded[expected_features]

# Convert to numpy array for compatibility
input_array = input_encoded.to_numpy()

# Prediction
if st.sidebar.button("Predict Loan Status"):
    if input_array.shape[1] != len(expected_features):
        st.error(f"Feature mismatch: expected {len(expected_features)} but got {input_array.shape[1]}.")
        st.stop()

    prediction = model.predict(input_array)
    prediction_proba = model.predict_proba(input_array)[0][1]

    if prediction[0] == 1:
        st.success(f"\u2705 Loan Approved (Probability: {prediction_proba:.2%})")
    else:
        st.error(f"\u274C Loan Not Approved (Probability: {prediction_proba:.2%})")
