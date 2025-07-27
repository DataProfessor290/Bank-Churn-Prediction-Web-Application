import streamlit as st
import pandas as pd
import joblib
import os

# ===============================
# 🎯 Load Model and Check Existence
# ===============================
model_path_xgb = os.path.join("models", "Churn XGB Classifier Model.pkl")

if not os.path.exists(model_path_xgb):
    st.error("❌ XGBoost model file not found. Please place 'Churn XGB Classifier Model.pkl' in the 'models/' directory.")
    st.stop()

xgb_model = joblib.load(model_path_xgb)

# ===============================
# 🧠 Define Prediction Function
# ===============================
def predict_churn(model, input_df):
    prediction = model.predict(input_df)
    return "🚪 Will Exit" if prediction[0] == 1 else "✅ Will Stay"

# ===============================
# 🎨 Streamlit App Config
# ===============================
st.set_page_config(
    page_title="Bank Customer Retention Predictor",
    page_icon="🏦",
    layout="centered"
)

# ===============================
# 🧭 Title and Introduction
# ===============================
st.markdown(
    """
    <h1 style='text-align: center; color: white;'>🏦 Bank Customer Retention Predictor</h1>
    <p style='text-align: center; color: #BBBBBB; font-size: 18px;'>
        Predict if a customer is likely to leave the bank using Machine Learning (XGBoost).
    </p>
    """,
    unsafe_allow_html=True
)

# ===============================
# 📋 Input Form
# ===============================
with st.form("churn_form"):
    st.subheader("📄 Enter Customer Information")

    col1, col2 = st.columns(2)

    with col1:
        geography = st.selectbox("🌍 Country", ["France", "Spain", "Germany"], help="Customer's country of residence")
        gender = st.selectbox("👤 Gender", ["Male", "Female"], help="Select customer's gender")
        age = st.slider("🎂 Age", 18, 92, 35, help="Select age between 18 and 92")
        tenure = st.slider("📆 Tenure (Years)", 0, 10, 3, help="Years with the bank")

    with col2:
        num_of_products = st.selectbox("📦 Number of Products", [1, 2, 3, 4], help="How many bank products the customer uses")
        has_cr_card = st.radio("💳 Has Credit Card?", ["Yes", "No"], help="Does the customer own a credit card?")
        balance = st.number_input("💰 Account Balance", min_value=0.0, value=50000.0, format="%.2f", help="Customer's account balance")
        est_salary = st.number_input("🧾 Estimated Salary", min_value=0.0, value=60000.0, format="%.2f", help="Estimated annual salary")

    submitted = st.form_submit_button("🔍 Predict")

# ===============================
# 🚀 Make Prediction
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
    st.subheader("📊 Prediction Result")

    result = predict_churn(xgb_model, input_data)
    st.success(f"⚡ XGBoost Model Prediction: **{result}**")

    st.markdown("---")

# ===============================
# 📫 Footer with Contact Info
# ===============================
st.markdown(
    """
    <hr style="border: 0.5px solid #444;" />
    <div style='text-align: center; color: gray; font-size: 14px;'>
        Made with ❤️ by <b>Tolulope Emuleomo</b><br/>
        📧 <a href="mailto:tolulopeemuleomo@gmail.com" style="color: lightgray;">tolulopeemuleomo@gmail.com</a> |
        💼 <a href="https://www.linkedin.com/in/tolulope-emuleomo" target="_blank" style="color: lightgray;">LinkedIn</a> |
        🐙 <a href="https://github.com/tolulopeemuleomo" target="_blank" style="color: lightgray;">GitHub</a><br/>
        <small>© 2025 Bank Retention Predictor App</small>
    </div>
    """,
    unsafe_allow_html=True
)
