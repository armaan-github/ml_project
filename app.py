import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import shap
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Customer Churn Prediction Dashboard", layout="wide")
st.title("Customer Churn Prediction App")
st.markdown("""
This app predicts customer churn using an XGBoost model and explains predictions with SHAP. Upload your customer data CSV to get started.
""")

uploaded_file = st.file_uploader("Upload your customer data CSV", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("### Data Preview", data.head())

    # Basic preprocessing
    data = data.dropna()  # Handle missing values (simple drop for demo)
    target_col = 'Churn'  # Change if your target column is named differently
    if target_col not in data.columns:
        st.error(f"Target column '{target_col}' not found in data.")
    else:
        X = data.drop(target_col, axis=1)
        y = data[target_col]
        # Encode target column to numeric (0/1)
        if y.dtype == 'O' or y.dtype.name == 'category':
            y = y.map({'No': 0, 'Yes': 1})
        if y.isnull().any():
            st.error("Target column contains values other than 'Yes'/'No'. Please check your data.")
        else:
            # Encode categorical features if any
            X = pd.get_dummies(X)
            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            # SMOTE for class imbalance
            smote = SMOTE(random_state=42)
            X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
            # Train XGBoost
            model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
            model.fit(X_train_res, y_train_res)
            # Predict
            y_pred = model.predict(X_test_scaled)
            st.write("### Classification Report", classification_report(y_test, y_pred, output_dict=True))
            # SHAP explanations
            explainer = shap.Explainer(model, X_train_res)
            shap_values = explainer(X_test_scaled)
            st.write("### Churn Predictions", pd.DataFrame({'Prediction': y_pred}, index=X_test.index))
            # Filter high-risk customers
            high_risk = X_test[y_pred == 1]
            st.write("### High-Risk Customers", high_risk)
            # SHAP plots
            st.write("### SHAP Value Plot for First Prediction")
            shap.plots.waterfall(shap_values[0], show=False)
            st.pyplot(plt.gcf())
            # Instructions for business users
            st.info("Use the filters above to focus on high-risk customers. SHAP plots explain why the model made each prediction.")
else:
    st.info("Please upload a CSV file to begin.")
