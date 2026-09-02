# Customer Churn Prediction using Logistic Regression

## Project Overview

This project predicts whether a bank customer is likely to churn using **Logistic Regression**.

The project uses customer demographic, financial, and account-related information to build a binary classification model.

### Objective

The main objective is to identify customers who are likely to leave the bank and understand which customer attributes have a significant impact on churn prediction.

## Dataset

The dataset contains **10,000 bank customer records** and includes information such as:

* Credit Score
* Country
* Gender
* Age
* Tenure
* Balance
* Number of Products
* Credit Card
* Active Member
* Estimated Salary
* Churn

`customer_id` was excluded from the model because it is an identifier and does not provide meaningful predictive information.

## Machine Learning Approach

The project follows these steps:

1. Load the customer churn dataset
2. Perform initial data inspection
3. Check data types
4. Check for missing values
5. Analyze the target variable distribution
6. Separate target and input features
7. Identify numerical and categorical features
8. Split the dataset into training and testing data
9. Scale numerical features using `RobustScaler`
10. Encode categorical features using `OneHotEncoder`
11. Train a Logistic Regression model
12. Evaluate model performance
13. Analyze Logistic Regression coefficients
14. Select important features
15. Train the model again using selected features
16. Save the trained model using Joblib
17. Use the saved model for future customer churn predictions

## Feature Selection

After analyzing the Logistic Regression coefficients, the following features were selected for the final model:

* Age
* Balance
* Country
* Gender
* Active Member

These features were selected based on their relative contribution in the trained Logistic Regression model.

## Model Evaluation

The final Logistic Regression model achieved:

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 77.53% |
| Precision | 59.78% |
| Recall    | 22.66% |
| F1 Score  | 32.87% |

Because customer churn is an imbalanced classification problem, accuracy alone is not sufficient to evaluate the model. Precision, recall, and F1-score were also considered.

## Model Deployment / Prediction

The trained model, scaler, and encoder are saved using `joblib`.

The saved model can later be loaded and used to enter new customer information and predict whether the customer is likely to:

* **Churn**
* **Not Churn**

## Technologies Used

* Python
* Pandas
* Scikit-learn
* Joblib
* Jupyter Notebook

## Machine Learning Algorithm

**Logistic Regression**

Logistic Regression is a supervised classification algorithm used here to predict a binary target variable:

* `0` → Customer will not churn
* `1` → Customer will churn

## Project Structure

customer-churn-logistic-regression/
│
├── Bank_Customer_Churn.csv
├── customer_churn_logistic_regression.ipynb
├── churn_model.joblib
├── predict.py
└── README.md

## Key Learning Outcomes

Through this project, I worked with:

* Data preprocessing
* Train-test split
* Numerical feature scaling
* Categorical feature encoding
* Logistic Regression
* Classification evaluation metrics
* Feature importance using model coefficients
* Model serialization using Joblib
* Making predictions using a saved machine learning model

