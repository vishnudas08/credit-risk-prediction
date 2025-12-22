
import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load the model
model = joblib.load("model/model.pkl")
scaler = joblib.load("model/scaler.pkl")
feature_cols = joblib.load("model/feature_cols.pkl")

st.title("💳 Credit Risk Prediction System")
st.write("Enter applicant details to predict loan default risk.")

# ------------------ Input Fields ------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Personal Information")
    person_age = st.number_input("Age", min_value=18, max_value=100, value=30)
    person_income = st.number_input("Annual Income ($)", min_value=0, value=50000)
    person_emp_length = st.number_input("Employment Length (years)", min_value=0.0, max_value=50.0, value=5.0)
    home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE"])
    cred_hist_length = st.number_input("Credit History Length (years)", min_value=0.0, max_value=50.0, value=5.0)
    default_history = st.selectbox("Previous Default on File", ["N", "Y"])

with col2:
    st.subheader("Loan Information")
    loan_amnt = st.number_input("Loan Amount ($)", min_value=0, value=10000)
    loan_int_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=30.0, value=10.0)
    loan_percent_income = st.number_input("Loan as % of Income", min_value=0.0, max_value=1.0, value=0.2, step=0.01)
    loan_intent = st.selectbox("Loan Intent", ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL"])
    loan_grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F"])

# ------------------ Derived Features ------------------
debt_to_income = loan_amnt / max(person_income, 1)
income_to_loan = person_income / max(loan_amnt, 1)
emp_stability = 1 if person_emp_length >= 2 else 0
high_interest = 1 if loan_int_rate > 15 else 0
high_utilization = 1 if debt_to_income > 0.4 else 0

# Age group encoding (must match training)
if person_age < 25:
    age_group_encoded = 0
elif person_age < 40:
    age_group_encoded = 1
elif person_age < 60:
    age_group_encoded = 2
else:
    age_group_encoded = 3

# ------------------ Encoding Maps ------------------
home_map = {"RENT": 0, "OWN": 1, "MORTGAGE": 2}
intent_map = {"EDUCATION": 0, "MEDICAL": 1, "VENTURE": 2, "PERSONAL": 3}
grade_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5}
default_map = {"N": 0, "Y": 1}

# ------------------ Prediction ------------------
if st.button("Predict Credit Risk"):
    input_dict = {
        "person_age": person_age,
        "person_income": person_income,
        "person_emp_length": person_emp_length,
        "loan_amnt": loan_amnt,
        "loan_int_rate": loan_int_rate,
        "loan_percent_income": loan_percent_income,
        "cb_person_cred_hist_length": cred_hist_length,
        "debt_to_income": debt_to_income,
        "income_to_loan": income_to_loan,
        "emp_stability": emp_stability,
        "high_interest": high_interest,
        "high_utilization": high_utilization,
        "person_home_ownership_encoded": home_map[home_ownership],
        "loan_intent_encoded": intent_map[loan_intent],
        "loan_grade_encoded": grade_map[loan_grade],
        "cb_person_default_on_file_encoded": default_map[default_history],
        "age_group_encoded": age_group_encoded
    }

    # VERY IMPORTANT: enforce correct column order
    input_data = pd.DataFrame([input_dict])[feature_cols]

    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High Risk of Default (Probability: {probability:.2%})")
    else:
        st.success(f"✅ Low Risk (Probability of Default: {probability:.2%})")