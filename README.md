# Customer Churn Analysis Report

This report presents the exploratory data analysis and data preparation steps undertaken to predict customer churn using various data sets. It includes data collection, visualisation, and cleaning processes, providing a foundation for developing a predictive model.

### Important note :
This is a job simulation involving customer churn prediction for the Data Science & Analytics team at Lloyds Banking Group
 
## Data Identification and Collection

The following data sets were identified as relevant for predicting customer churn:
- Customer Demographics
- Transaction History
- Customer Service Interactions
- Online Activity
- Churn Status
These data sets provide comprehensive insights into customer behaviour and engagement.

## Exploratory Data Analysis

Exploratory data analysis was conducted to uncover patterns and insights that could inform predictive modelling. Key findings include demographic trends, spending patterns, and the impact of customer service interactions on churn.

## Data Cleaning and Preparation

Data cleaning involved handling missing values, detecting and addressing outliers. Categorical features were encoded using get_dummies method to prepare the data for machine learning algorithms.

## Predictive Data Model

Developed and implemented a predictive model using Decision Tree and Random Forest machine learning algorithms, achieving an Accuracy score of 0.97, with a recall value for class 0 as 1.00, while for clas 1 as 0.84. for class being very important for predicting the churn, i went futher to improve recall for class using Tune classification threshold, which increased the recall value from 0.84 to 0.93 and accuracy value from 0.97 to 0.98.

