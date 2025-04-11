# 🏦 Bank Churn Prediction Web App

An interactive Streamlit web application for predicting bank customer churn using various machine learning algorithms.

---

## 📌 Project Overview

This project aims to provide a predictive system that helps banks identify customers likely to churn (leave the bank). It allows users to explore and evaluate different classification models and predict customer churn based on specific input features like geography, tenure, number of products, and estimated salary.

---

## 🔧 Features

- ✅ Clean and intuitive Streamlit user interface  
- 🔍 Model evaluation: Compare multiple models using F1 Score, confusion matrix, and classification report  
- 🔮 Predict churn for a new customer with selected model  
- 📊 Handles imbalanced dataset with undersampling  
- 🔁 Preprocessing pipeline using Scikit-learn transformers  

---

## 🧠 Machine Learning Models Used

- Random Forest  
- Support Vector Machine (SVM)  
- K-Nearest Neighbors (KNN)  
- Decision Tree  
- Gradient Boosting  
- XGBoost  

---

## 🛠️ Tech Stack

- **Programming Language**: Python  
- **Web Framework**: Streamlit  
- **ML Libraries**: Scikit-learn, XGBoost, imbalanced-learn  
- **Data Manipulation**: Pandas, Numpy  

---

## 📁 Dataset

The dataset used is a sample bank churn dataset in CSV format containing customer demographic and account information.

**Features used:**
- `Geography`
- `Tenure`
- `Number of Products`
- `Estimated Salary`

**Target variable:**
- `Exited` (1 = churned, 0 = not churned)
