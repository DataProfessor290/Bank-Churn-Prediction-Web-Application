import streamlit as st
import pandas as pd
import joblib
from PIL import Image

# ===============================
# 🎯 Load Model and Assets
# ===============================
rf_model = joblib.load("Churn RF Classifier Model.pkl")
xgb_model = joblib.load("Churn XGB Classifier Model.pkl")

# ===============================
# 🧠 Define Prediction Function
# ===============================
def predict_churn(model, input_df):
    prediction = model.predict(input_df)
    result = "Will Exit" if prediction[0] == 1 else "Will Stay"
    return result

# ===============================
# 🎨 Streamlit App Layout
# ===============================
st.set_page_config(page_title="Bank Churn Prediction App", page_icon="🩺", layout="centered")

st.markdown(
    """
    <h1 style='text-align: center; color: #FAFAFA;'>🏦 Bank Churn Prediction</h1>
    <p style='text-align: center; color: #A9A9A9;'>🔍 Know your customer. Predict whether a client is likely to churn using key features and ML models.</p>
    """,
    unsafe_allow_html=True
)

with st.form("churn_form"):
    st.subheader("📋 Customer Details")

    col1, col2 = st.columns(2)

    with col1:
        geography = st.selectbox("🌍 Geography", ["France", "Spain", "Germany"])
        gender = st.selectbox("🧑 Gender", ["Male", "Female"])
        age = st.slider("🎂 Age", 18, 92, 35)
        tenure = st.slider("📆 Tenure (Years)", 0, 10, 5)

    with col2:
        num_of_products = st.selectbox("📦 Number of Products", [1, 2, 3, 4])
        has_cr_card = st.radio("💳 Has Credit Card?", ["Yes", "No"])
        balance = st.number_input("💰 Balance", min_value=0.0, value=50000.0, format="%.2f")
        est_salary = st.number_input("🧾 Estimated Salary", min_value=0.0, value=60000.0, format="%.2f")

    submit = st.form_submit_button("Predict")

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
        st.success(f"🌲 Random Forest: **{rf_result}**")

    with col2:
        xgb_result = predict_churn(xgb_model, input_data)
        st.success(f"⚡ XGBoost: **{xgb_result}**")

    st.markdown("---")

# ===============================
# 🦶 Footer
# ===============================
st.markdown(
    """
    <hr style="border: 0.5px solid #4B4B4B;"/>
    <p style='text-align: center; font-size: 14px; color: gray'>
        © 2025 Data Professor | Built with ❤️ by Tolulope Emuleomo | Bank Churn Prediction Web App.
    </p>
    """,
    unsafe_allow_html=True
) 
