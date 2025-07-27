import streamlit as st
import pandas as pd
import joblib
import os
from PIL import Image

# ===============================
# 🎯 Load Model and Assets
# ===============================
model_path_rf = os.path.join("models", "Churn RF Classifier Model.pkl")
model_path_xgb = os.path.join("models", "Churn XGB Classifier Model.pkl")

if not os.path.exists(model_path_rf) or not os.path.exists(model_path_xgb):
    st.error("❌ Model files not found. Please ensure both model files exist in the 'models/' folder.")
    st.stop()

rf_model = joblib.load(model_path_rf)
xgb_model = joblib.load(model_path_xgb)

# ===============================
# 🧠 Define Prediction Function
# ===============================
def predict_churn(model, input_df):
    prediction = model.predict(input_df)
    return "🚪 Will Exit" if prediction[0] == 1 else "✅ Will Stay"

# ===============================
# 🎨 Streamlit App Layout
# ===============================
st.set_page_config(
    page_title="Bank Customer Retention Predictor",
    page_icon="🏦",
    layout="centered",
    initial_sidebar_state="auto"
)

st.markdown(
    """
    <h1 style='text-align: center; color: white;'>🏦 Bank Customer Retention Predictor</h1>
    <p style='text-align: center; color: #BBBBBB; font-size: 18px;'>Use machine learning to predict whether a customer is likely to churn based on key features.</p>
    """,
    unsafe_allow_html=True
)

# ===============================
# 📋 Input Form
# ===============================
with st.form("churn_form"):
    st.subheader("📄 Customer Details")

    col1, col2 = st.columns(2)

    with col1:
        geography = st.selectbox("🌍 Customer's Country", ["France", "Spain", "Germany"], help="Select customer's country of residence")
        gender = st.selectbox("👤 Gender", ["Male", "Female"], help="Select the gender of the customer")
        age = st.slider("🎂 Age", 18, 92, 35, help="Select customer's age (18-92)")
        tenure = st.slider("📆 Tenure (years)", 0, 10, 5, help="Number of years the customer has been with the bank")

    with col2:
        num_of_products = st.selectbox("📦 Number of Products", [1, 2, 3, 4], help="How many bank products the customer uses")
        has_cr_card = st.radio("💳 Has Credit Card?", ["Yes", "No"], help="Does the customer have a credit card?")
        balance = st.number_input("💰 Balance", min_value=0.0, value=50000.0, format="%.2f", help="Current balance in customer's account")
        est_salary = st.number_input("🧾 Estimated Salary", min_value=0.0, value=60000.0, format="%.2f", help="Customer's estimated yearly salary")

    submit = st.form_submit_button("🔍 Predict")

# ===============================
# 🚀 Make Prediction
# ===============================
if submit:
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
    st.subheader("🤖 Model Predictions")

    col1, col2 = st.columns(2)

    with col1:
        rf_result = predict_churn(rf_model, input_data)
        st.success(f"🌲 Random Forest Prediction: **{rf_result}**")

    with col2:
        xgb_result = predict_churn(xgb_model, input_data)
        st.success(f"⚡ XGBoost Prediction: **{xgb_result}**")

    st.markdown("---")

# ===============================
# 📫 Footer
# ===============================
st.markdown(
    """
    <hr style="border: 0.5px solid #444;" />
    <div style='text-align: center; color: gray; font-size: 14px;'>
        Developed with ❤️ by <b>Tolulope Emuleomo</b><br/>
        📧 <a href="mailto:tolulopeemuleomo@gmail.com" style="color: lightgray;">tolulopeemuleomo@gmail.com</a> |
        💼 <a href="https://www.linkedin.com/in/tolulope-emuleomo" target="_blank" style="color: lightgray;">LinkedIn</a> |
        🐙 <a href="https://github.com/tolulopeemuleomo" target="_blank" style="color: lightgray;">GitHub</a><br/>
        <small>© 2025 Bank Retention Predictor App</small>
    </div>
    """,
    unsafe_allow_html=True
)
