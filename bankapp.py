import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Set wide layout and page config
st.set_page_config(page_title="📊 Bank Churn Predictor", page_icon="💳", layout="wide")
st.markdown("""
    <style>
        .main {background-color: #F0F2F6;}
        footer {visibility: hidden;}
        .footer:after {
            content:'\00a9 2025 Powered by Data Professor. Built with ❤️ using Streamlit';
            visibility: visible;
            display: block;
            position: relative;
            text-align: center;
            padding: 10px;
            color: gray;
        }
    </style>
    <div class="footer"></div>
""", unsafe_allow_html=True)

# Title
st.title("💳 Bank Churn Prediction App")
st.markdown("Predict whether a customer is likely to churn using top-performing machine learning models.")

# Load models
xgb_model = joblib.load("Churn XGB Classifier Model.pkl")
rf_model = joblib.load("Churn RF Classifier Model.pkl")

# User input
st.sidebar.header("📥 Input Customer Details")
credit_score = st.sidebar.number_input("Credit Score", min_value=300, max_value=900, value=650, help="Customer's credit score")
country = st.sidebar.selectbox("Country", ["France", "Germany", "Spain"], help="Customer's geography")
gender = st.sidebar.selectbox("Gender", ["Male", "Female"], help="Customer's gender")
age = st.sidebar.slider("Age", 18, 100, 40, help="Customer's age")
tenure = st.sidebar.slider("Tenure (Years)", 0, 10, 3, help="Number of years the customer has been with the bank")
balance = st.sidebar.number_input("Account Balance", value=125000.00, help="Customer's account balance")
num_of_products = st.sidebar.selectbox("Number of Products", [1, 2, 3, 4], help="Number of bank products the customer uses")
has_cr_card = st.sidebar.selectbox("Has Credit Card?", ["Yes", "No"], help="Does the customer have a credit card?")
is_active_member = st.sidebar.selectbox("Active Member?", ["Yes", "No"], help="Is the customer an active member?")
estimated_salary = st.sidebar.number_input("Estimated Salary", value=100000.00, help="Estimated annual salary of the customer")

# Prepare input for prediction
input_data = pd.DataFrame({
    "credit_score": [credit_score],
    "country": [country],
    "gender": [gender],
    "age": [age],
    "tenure": [tenure],
    "balance": [balance],
    "num_of_products": [num_of_products],
    "has_cr_card": [1 if has_cr_card == "Yes" else 0],
    "is_active_member": [1 if is_active_member == "Yes" else 0],
    "estimated_salary": [estimated_salary]
})

st.subheader("📄 Input Summary")
st.write(input_data)

# Prediction button
if st.button("📊 Predict Churn"):
    prediction_xgb = xgb_model.predict(input_data)[0]
    prediction_rf = rf_model.predict(input_data)[0]

    label = "❌ Likely to Churn" if prediction_xgb == 1 else "✅ Not Likely to Churn"

    st.markdown("---")
    st.subheader("🔍 Prediction Result (XGBoost)")
    st.success(label) if prediction_xgb == 0 else st.error(label)

    st.subheader("🌲 Prediction Result (Random Forest)")
    st.success("✅ Not Likely to Churn") if prediction_rf == 0 else st.error("❌ Likely to Churn")

    st.markdown("---")
    st.markdown("✔️ XGBoost is the primary model used due to its better performance.")

# Footer (already styled in markdown above)
