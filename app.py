import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load your model
model = joblib.load("customer_churn_model.pkl")

st.title("Customer Churn Prediction App")

# Tabs for input type
tab1, tab2 = st.tabs(["🔢 Manual Entry", "📂 Upload CSV"])

# ========== TAB 1: Manual Input ==========
with tab1:
    st.subheader("Enter Customer Information")

    age = st.number_input("Age", min_value=0)
    amount_spent = st.number_input("Amount Spent", min_value=0.0)
    login_frequency = st.number_input("Login Frequency", min_value=0)
    transaction_year = st.number_input("Transaction Year", min_value=2000, max_value=2100)
    transaction_month = st.number_input("Transaction Month", min_value=1, max_value=12)
    transaction_day = st.number_input("Transaction Day", min_value=1, max_value=31)
    last_login_year = st.number_input("Last Login Year", min_value=2000, max_value=2100)
    last_login_month = st.number_input("Last Login Month", min_value=1, max_value=12)
    last_login_day = st.number_input("Last Login Day", min_value=1, max_value=31)

    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital Status", ["Married", "Single", "Widowed"])
    income_level = st.selectbox("Income Level", ["Low", "Medium", "High"])
    interaction_type = st.selectbox("Interaction Type", ["Feedback", "Inquiry", "Unknown"])
    resolution_status = st.selectbox("Resolution Status", ["Unknown", "Unresolved", "Resolved"])
    product_category = st.selectbox("Product Category", ["Clothing", "Electronics", "Furniture", "Groceries", "Other"])
    service_usage = st.multiselect("Service Usage", ["Online Banking", "Website"])

    # One-hot encoding
    gender_m = 1 if gender == "Male" else 0
    married = 1 if marital_status == "Married" else 0
    single = 1 if marital_status == "Single" else 0
    widowed = 1 if marital_status == "Widowed" else 0
    income_low = 1 if income_level == "Low" else 0
    income_medium = 1 if income_level == "Medium" else 0
    interaction_feedback = 1 if interaction_type == "Feedback" else 0
    interaction_inquiry = 1 if interaction_type == "Inquiry" else 0
    interaction_unknown = 1 if interaction_type == "Unknown" else 0
    res_unknown = 1 if resolution_status == "Unknown" else 0
    res_unresolved = 1 if resolution_status == "Unresolved" else 0
    prod_clothing = 1 if product_category == "Clothing" else 0
    prod_electronics = 1 if product_category == "Electronics" else 0
    prod_furniture = 1 if product_category == "Furniture" else 0
    prod_groceries = 1 if product_category == "Groceries" else 0
    service_online = 1 if "Online Banking" in service_usage else 0
    service_web = 1 if "Website" in service_usage else 0

    # Create input feature array
    features = np.array([[
        age, amount_spent, login_frequency, transaction_year, transaction_month, transaction_day,
        last_login_year, last_login_month, last_login_day,
        gender_m, married, single, widowed,
        income_low, income_medium,
        interaction_feedback, interaction_inquiry, interaction_unknown,
        res_unknown, res_unresolved,
        prod_clothing, prod_electronics, prod_furniture, prod_groceries,
        service_online, service_web
    ]])

    if st.button("Predict Churn (Manual)"):
        pred = model.predict(features)[0]
        label = "Churn" if pred == 1 else "No Churn"
        st.success(f"The model predicts: *{label}*")


# ========== TAB 2: CSV Upload ==========
with tab2:
    st.subheader("Upload CSV File")

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        st.write("📊 Uploaded Data Preview")
        st.dataframe(df.head())

        try:
            # Ensure all required columns are present
            expected_columns = [
                'Age', 'AmountSpent', 'LoginFrequency', 'TransactionYear', 'TransactionMonth',
                'TransactionDay', 'LastLoginYear', 'LastLoginMonth', 'LastLoginDay',
                'Gender_M', 'MaritalStatus_Married', 'MaritalStatus_Single', 'MaritalStatus_Widowed',
                'IncomeLevel_Low', 'IncomeLevel_Medium',
                'InteractionType_Feedback', 'InteractionType_Inquiry', 'InteractionType_Unknown',
                'ResolutionStatus_Unknown', 'ResolutionStatus_Unresolved',
                'ProductCategory_Clothing', 'ProductCategory_Electronics', 'ProductCategory_Furniture',
                'ProductCategory_Groceries', 'ServiceUsage_Online Banking', 'ServiceUsage_Website'
            ]

            missing = [col for col in expected_columns if col not in df.columns]
            if missing:
                st.error(f"🚫 The following required columns are missing: {missing}")
            else:
                preds = model.predict(df[expected_columns])
                df['Prediction'] = ["Churn" if x == 1 else "No Churn" for x in preds]

                st.success("✅ Predictions completed!")
                st.dataframe(df[['Prediction']].join(df))

                csv_download = df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Predictions", data=csv_download, file_name="churn_predictions.csv", mime="text/csv")

        except Exception as e:
            st.error(f"Error during prediction: {e}")
