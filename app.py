import streamlit as st 
import joblib 
import numpy as np

#loading my trained model
model = joblib.load("Customer_churn_model_compressed.pkl")

st.title("Customer Churn Prediction App")
st.write("Enter customer details to predict if they wil churn.")

#model input (Requred features)
# Numerical inputs
age = st.number_input("Age", min_value=0)
amount_spent = st.number_input("Amount Spent", min_value=0.0)
login_frequency = st.number_input("Login Frequency", min_value=0)
transaction_year = st.number_input("Transaction Year", min_value=2000, max_value=2100)
transaction_month = st.number_input("Transaction Month", min_value=1, max_value=12)
transaction_day = st.number_input("Transaction Day", min_value=1, max_value=31)
last_login_year = st.number_input("Last Login Year", min_value=2000, max_value=2100)
last_login_month = st.number_input("Last Login Month", min_value=1, max_value=12)
last_login_day = st.number_input("Last Login Day", min_value=1, max_value=31)

# Categorical inputs (one-hot encoded manually)
gender = st.selectbox("Gender", ["Male", "Female"])
marital_status = st.selectbox("Marital Status", ["Married", "Single", "Widowed"])
income_level = st.selectbox("Income Level", ["Low", "Medium", "High"])
interaction_type = st.selectbox("Interaction Type", ["Feedback", "Inquiry", "Unknown"])
resolution_status = st.selectbox("Resolution Status", ["Unknown", "Unresolved", "Resolved"])
product_category = st.selectbox("Product Category", ["Clothing", "Electronics", "Furniture", "Groceries", "Other"])
service_usage = st.multiselect("Service Usage", ["Online Banking", "Website"])

# One-hot encode categorical variables
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

# Combine features into the correct input order
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

# Predict
if st.button("Predict Churn"):
    prediction = model.predict(features)[0]
    label = "Churn" if prediction == 1 else "No Churn"
    st.success(f"The model predicts: *{label}*")
