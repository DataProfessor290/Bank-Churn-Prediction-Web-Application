# Importing required libraries
import streamlit as st
import pandas as pd
import warnings

# Machine learning libraries
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from imblearn.under_sampling import RandomUnderSampler

# Required for background_gradient styling in classification report
import matplotlib.pyplot as plt

# Ignore warnings to keep the interface clean
warnings.filterwarnings("ignore")

# ------------------- DATA LOADING AND PREPARATION -------------------

# Cache data loading for performance
@st.cache_data
def load_data():
    # Load dataset from CSV and standardize column names
    df = pd.read_csv("bank_churn_data.csv")
    df.columns = df.columns.str.lower()  # Convert all column names to lowercase
    df['customerid'] = df['customerid'].astype('object')  # Convert ID fields to categorical
    df['id'] = df['id'].astype('object')
    return df

# Function to split columns by type: numerical or categorical
def split_cols(data):
    num_cols = data.select_dtypes(include=['number']).columns
    cat_cols = data.select_dtypes(include=['object']).columns
    return num_cols, cat_cols

# Prepares and preprocesses the dataset for modeling
def prepare_data(df):
    # Select features and target
    features = ['tenure', 'numofproducts', 'geography', 'estimatedsalary']
    X = df[features]
    y = df.exited

    # Balance the dataset using undersampling
    sampler = RandomUnderSampler()
    X_resampled, y_resampled = sampler.fit_resample(X, y)

    # Identify numerical and categorical columns
    num_cols, cat_cols = split_cols(X_resampled)

    # Define preprocessing pipelines for numerical and categorical features
    num_pipe = Pipeline([('scaler', StandardScaler())])
    cat_pipe = Pipeline([('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))])

    # Combine preprocessing into a column transformer
    processor = ColumnTransformer([
        ('num', num_pipe, num_cols),
        ('cat', cat_pipe, cat_cols)
    ], remainder='passthrough')  # Pass through any remaining columns

    # Split data into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

    return processor, x_train, x_test, y_train, y_test, features

# Returns a dictionary of machine learning models wrapped in pipelines
def get_models(processor):
    base_models = {
        'Random Forest': RandomForestClassifier(),
        'SVM': SVC(),
        'KNN': KNeighborsClassifier(),
        'Decision Tree': DecisionTreeClassifier(),
        'Gradient Boosting': GradientBoostingClassifier(),
        'XGBoost': XGBClassifier()
    }

    # Each model is paired with the preprocessor in a pipeline
    return {
        name: Pipeline([
            ('processor', processor),
            ('model', model)
        ]) for name, model in base_models.items()
    }

# ------------------- STREAMLIT WEB APP -------------------

# Set the app title and layout
st.set_page_config(page_title="Bank Churn App", layout="wide")
st.title("🏦 Bank Churn Prediction App")

# Load and preprocess the data
df = load_data()
processor, x_train, x_test, y_train, y_test, features = prepare_data(df)
models = get_models(processor)

# Sidebar navigation for choosing between pages
st.sidebar.title("📍 Navigation")
page = st.sidebar.radio("Go to", ["Model Evaluation", "Predict New User"])

# ------------------- PAGE 1: MODEL EVALUATION -------------------

if page == "Model Evaluation":
    st.subheader("📊 Evaluate Selected Models")

    # Sidebar for selecting which models to evaluate
    st.sidebar.markdown("### 🤖 Choose Models to Evaluate")
    selected_models = st.sidebar.multiselect("Select Models", options=list(models.keys()), default=list(models.keys())[:2])
    run_eval = st.sidebar.button("Run Evaluation")  # Button to trigger evaluation

    if run_eval:
        if not selected_models:
            st.warning("Please select at least one model to evaluate.")
        else:
            st.info("Training selected models and generating reports...")

            eval_results = {}  # Dictionary to store evaluation results

            for name in selected_models:
                model = models[name]
                with st.spinner(f"Training {name}..."):  # Show spinner while training
                    trained_model = model.fit(x_train, y_train)
                    preds = trained_model.predict(x_test)

                    # Get F1 score, classification report, and confusion matrix
                    f1 = f1_score(y_test, preds)
                    report = classification_report(y_test, preds, output_dict=True)
                    matrix = confusion_matrix(y_test, preds)

                    # Store results
                    eval_results[name] = {
                        'f1': f1,
                        'report': report,
                        'matrix': matrix
                    }

            # Show results in tabs, one for each model
            tabs = st.tabs(list(eval_results.keys()))
            for i, name in enumerate(eval_results.keys()):
                with tabs[i]:
                    st.markdown(f"### {name} Evaluation")
                    st.metric("F1 Score", f"{eval_results[name]['f1']:.4f}")  # Display F1 score

                    # Show classification report with color gradient
                    st.markdown("**Classification Report**")
                    styled_report = pd.DataFrame(eval_results[name]['report']).T.style.background_gradient(cmap="Blues")
                    st.dataframe(styled_report)

                    # Show confusion matrix
                    st.markdown("**Confusion Matrix**")
                    st.write(eval_results[name]['matrix'])

    else:
        st.info("Use the sidebar to choose models and click 'Run Evaluation'.")

# ------------------- PAGE 2: PREDICT NEW USER -------------------

else:
    st.subheader("🔮 Predict Churn for a New Customer")

    st.markdown("### 📥 Enter Customer Information")
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)  # Split form into two columns

        # Input: geography and tenure
        with col1:
            geography = st.selectbox("Geography", df["geography"].unique())
            tenure = st.slider("Tenure (Years)", 0, 10, 3)

        # Input: number of products and salary
        with col2:
            num_products = st.slider("Number of Products", 1, 4, 1)
            salary = st.number_input("Estimated Salary", value=50000.0, step=1000.0)

        # Choose model for prediction
        model_choice = st.selectbox("Choose a Model", list(models.keys()))
        submit = st.form_submit_button("Predict")  # Submit button

    # On form submission
    if submit:
        # Collect input data into a DataFrame
        input_data = pd.DataFrame({
            "geography": [geography],
            "tenure": [tenure],
            "numofproducts": [num_products],
            "estimatedsalary": [salary]
        })

        # Train and use selected model for prediction
        model = models[model_choice].fit(x_train, y_train)
        prediction = model.predict(input_data)[0]

        # Show prediction result
        st.markdown("## 🎯 Prediction Result")
        st.write("Using Model:", f"**{model_choice}**")
        st.write("Customer Data:")
        st.dataframe(input_data)

        # Display prediction as churn or not churn
        if prediction == 1:
            st.error("❌ The customer is likely to **CHURN**.")
        else:
            st.success("✅ The customer is **NOT likely to churn**.")
