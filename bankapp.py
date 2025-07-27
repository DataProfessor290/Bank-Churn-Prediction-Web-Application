import streamlit as st
import pandas as pd
import warnings
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

# Suppress warnings to avoid clutter
warnings.filterwarnings("ignore")

# =============================
# 📦 Load & Cache Data
# =============================
@st.cache_data
def load_data():
    df = pd.read_csv("bank_churn_data.csv")
    df.columns = df.columns.str.lower()
    df['customerid'] = df['customerid'].astype('object')
    df['id'] = df['id'].astype('object')
    return df

# =============================
# 🔧 Data Preparation
# =============================
def split_cols(data):
    num_cols = data.select_dtypes(include=['number']).columns
    cat_cols = data.select_dtypes(include=['object']).columns
    return num_cols, cat_cols

def prepare_data(df):
    features = ['tenure', 'numofproducts', 'geography', 'estimatedsalary']
    X = df[features]
    y = df['exited']

    # Resample
    sampler = RandomUnderSampler()
    X_resampled, y_resampled = sampler.fit_resample(X, y)

    # Column split
    num_cols, cat_cols = split_cols(X_resampled)

    # Preprocessing pipelines
    num_pipe = Pipeline([('scaler', StandardScaler())])
    cat_pipe = Pipeline([('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))])

    processor = ColumnTransformer([
        ('num', num_pipe, num_cols),
        ('cat', cat_pipe, cat_cols)
    ], remainder='passthrough')

    x_train, x_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

    return processor, x_train, x_test, y_train, y_test, features

# =============================
# 🧠 Model Dictionary
# =============================
def get_models(processor):
    base_models = {
        'Random Forest': RandomForestClassifier(),
        'SVM': SVC(),
        'KNN': KNeighborsClassifier(),
        'Decision Tree': DecisionTreeClassifier(),
        'Gradient Boosting': GradientBoostingClassifier(),
        'XGBoost': XGBClassifier()
    }

    return {
        name: Pipeline([
            ('processor', processor),
            ('model', model)
        ]) for name, model in base_models.items()
    }

# =============================
# 🧠 Train All Models (Cached)
# =============================
@st.cache_resource
def train_models(models, x_train, y_train):
    for model in models.values():
        model.fit(x_train, y_train)
    return models

# =============================
# 🎯 Streamlit App UI
# =============================
st.set_page_config(page_title="Bank Churn App", layout="wide")
st.title("🏦 Bank Churn Prediction App")

df = load_data()
processor, x_train, x_test, y_train, y_test, features = prepare_data(df)
models = get_models(processor)
models = train_models(models, x_train, y_train)

# =============================
# 📍 Sidebar Navigation
# =============================
st.sidebar.title("📍 Navigation")
page = st.sidebar.radio("Go to", ["Model Evaluation", "Predict New User"])

# =============================
# 📊 Model Evaluation
# =============================
if page == "Model Evaluation":
    st.subheader("📊 Evaluate Selected Models")

    st.sidebar.markdown("### 🤖 Choose Models to Evaluate")
    selected_models = st.sidebar.multiselect("Select Models", options=list(models.keys()), default=list(models.keys())[:2])
    run_eval = st.sidebar.button("Run Evaluation")

    if run_eval:
        if not selected_models:
            st.warning("Please select at least one model to evaluate.")
        else:
            st.info("Training selected models and generating reports...")

            eval_results = {}
            for name in selected_models:
                model = models[name]
                with st.spinner(f"Evaluating {name}..."):
                    preds = model.predict(x_test)

                    f1 = f1_score(y_test, preds)
                    report = classification_report(y_test, preds, output_dict=True)
                    matrix = confusion_matrix(y_test, preds)

                    eval_results[name] = {
                        'f1': f1,
                        'report': report,
                        'matrix': matrix
                    }

            tabs = st.tabs(list(eval_results.keys()))
            for i, name in enumerate(eval_results.keys()):
                with tabs[i]:
                    st.markdown(f"### {name} Evaluation")
                    st.metric("F1 Score", f"{eval_results[name]['f1']:.4f}")

                    st.markdown("**Classification Report**")
                    report_df = pd.DataFrame(eval_results[name]['report']).T
                    report_df.rename(index={"0": "Not Exited", "1": "Exited"}, inplace=True)
                    st.dataframe(report_df)

                    st.markdown("**Confusion Matrix**")
                    matrix_df = pd.DataFrame(
                        eval_results[name]['matrix'],
                        index=["Actual Not Exited", "Actual Exited"],
                        columns=["Predicted Not Exited", "Predicted Exited"]
                    )
                    st.dataframe(matrix_df)
    else:
        st.info("Use the sidebar to choose models and click 'Run Evaluation'.")

# =============================
# 🔮 Predict New Customer
# =============================
else:
    st.subheader("🔮 Predict Churn for a New Customer")

    st.markdown("### 📥 Enter Customer Information")
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            geography = st.selectbox("Geography", df["geography"].unique())
            tenure = st.slider("Tenure (Years)", 0, 10, 3)

        with col2:
            num_products = st.slider("Number of Products", 1, 4, 1)
            salary = st.number_input("Estimated Salary", value=50000.0, step=1000.0)

        model_choice = st.selectbox("Choose a Model", list(models.keys()))
        submit = st.form_submit_button("Predict")

    if submit:
        input_data = pd.DataFrame({
            "geography": [geography],
            "tenure": [tenure],
            "numofproducts": [num_products],
            "estimatedsalary": [salary]
        })

        selected_model = models[model_choice]
        prediction = selected_model.predict(input_data)[0]

        st.markdown("## 🎯 Prediction Result")
        st.write("Using Model:", f"**{model_choice}**")
        st.write("Customer Data:")
        st.dataframe(input_data)

        if prediction == 1:
            st.error("❌ The customer is likely to **CHURN**.")
        else:
            st.success("✅ The customer is **NOT likely to churn**.")

# =============================
# 📝 Footer
# =============================
st.markdown("---")
st.markdown(
    """
    <div style="font-size: 13px; color: #999999; text-align: center;">
        Built with ❤️ using <a href="https://streamlit.io" target="_blank" style="color: #1f77b4;">Streamlit</a><br>
        Created by <strong>Tolulope Emuleomo</strong> aka <strong>Data Professor</strong> 🧠<br>
        🔗 <a href="https://twitter.com/dataprofessor_" target="_blank" style="color: #1DA1F2;">@dataprofessor_</a> |
        <a href="https://github.com/dataprofessor290" target="_blank" style="color: #6e5494;">GitHub</a><br>
        💼 <span style="color: #cccccc;">Data Scientist</span>
    </div>
    """,
    unsafe_allow_html=True
)
