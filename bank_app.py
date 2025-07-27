import streamlit as st
import pandas as pd
import joblib
import os

# ===============================
# 🎯 Load Models
# ===============================
try:
    rf_model = joblib.load("Churn RF Classifier Model.pkl")
    xgb_model = joblib.load("Churn XGB Classifier Model.pkl")
except FileNotFoundError as e:
    st.error(f"❌ Model file not found: {e.filename}")
    st.stop()

# ===============================
# 🧠 Define Prediction Function
# ===============================
def predict_churn(model, input_df):
    prediction = model.predict(input_df)
    return "💔 Will Exit" if prediction[0] == 1 else "💚 Will Stay"

# ===============================
# 🎨 Page Setup
# ===============================
st.set_page_config(
    page_title="Bank Churn Prediction App",
    page_icon="🏦",
    layout="centered"
)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    h1, h2, h3, h4, h5, h6, p {
        color: #FAFAFA;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ===============================
# 🏦 App Title
# ===============================
st.markdown(
    """
    <h1 style='text-align: center;'>🏦 Bank Churn Prediction</h1>
    <p style='text-align: center; color: #CCCCCC;'>Predict whether a customer will leave the bank using Machine Learning models trained on real data.</p>
    """,
    unsafe_allow_html=True
)

# ===============================
# 📋 Input Form
# ===============================
with st.form("churn_form"):
    st.subheader("🧾 Enter Customer Details")

    col1, col2 = st.columns(2)

    with col1:
        geography = st.selectbox("🌍 Customer's Country", ["France", "Spain", "Germany"], help="Select the country where the customer lives.")
        gender = st.selectbox("👤 Gender", ["Male", "Female"], help="Select the customer's gender.")
        age = st.slider("🎂 Age", 18, 92, 35, help="Select the customer's age.")
        tenure = st.slider("📆 Tenure", 0, 10, 5, help="How many years the customer has been with the bank.")

    with col2:
        num_of_products = st.selectbox("📦 Number of Products", [1, 2, 3, 4], help="How many bank products the customer uses.")
        has_cr_card = st.radio("💳 Has Credit Card?", ["Yes", "No"], help="Does the customer have a credit card?")
        balance = st.number_input("💰 Account Balance", min_value=0.0, value=50000.0, format="%.2f", help="Customer's current bank balance.")
        est_salary = st.number_input("🧾 Estimated Salary", min_value=0.0, value=60000.0, format="%.2f", help="Estimated yearly salary of the customer.")

    submitted = st.form_submit_button("🚀 Predict Churn")

# ===============================
# 🚀 Run Predictions
# ===============================
if submitted:
    input_data = pd.DataFrame([{
        "geography": geography,
        "gender": gender,
        "age": age,
        "tenure": tenure,
        "numofproducts": num_of_products,
        "hascrcard": 1 if has_cr_card == "Yes" else 0,
        "balance": balance,
        "estimatedsalary": est_salary
    }])

    st.markdown("---")
    st.subheader("🤖 Prediction Results")

    col1, col2 = st.columns(2)

    with col1:
        rf_result = predict_churn(rf_model, input_data)
        st.success(f"🌲 Random Forest: **{rf_result}**")

    with col2:
        xgb_result = predict_churn(xgb_model, input_data)
        st.success(f"⚡ XGBoost: **{xgb_result}**")

    st.markdown("---")

# ===============================
# 🧾 Footer with Contact Info
# ===============================
st.markdown(
    """
    <hr style="border: 0.5px solid #4B4B4B;"/>
    <div style='text-align: center; font-size: 14px; color: #BBBBBB;'>
        <p>Developed with ❤️ by <strong>Tolulope Emuleomo</strong></p>
        <p>
            📧 <a href='mailto:tolulope.emuleomo@gmail.com' style='color:#1E90FF;'>tolulope.emuleomo@gmail.com</a><br>
            🔗 
            <a href='https://www.linkedin.com/in/tolulope-emuleomo' target='_blank' style='margin: 0 10px; color:#1E90FF;'>LinkedIn</a> | 
            <a href='https://github.com/Tolulope-Emuleomo' target='_blank' style='margin: 0 10px; color:#1E90FF;'>GitHub</a>
        </p>
        <p style='font-size: 13px; color: gray;'>© 2025 Tolulope Emuleomo | Bank Churn Prediction App</p>
    </div>
    """,
    unsafe_allow_html=True
)
