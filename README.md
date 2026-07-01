# 💳 Credit Card Default Prediction using Machine Learning

## 📌 Project Overview

This project predicts whether a credit card customer is likely to default on their next month's payment using Machine Learning techniques. The goal is to assist financial institutions in identifying high-risk customers and improving credit risk management.

The project follows a complete end-to-end Machine Learning workflow including data preprocessing, exploratory data analysis (EDA), feature engineering, model development, explainability using SHAP, and deployment using Streamlit.

---

## 🎯 Problem Statement

Credit card default poses a significant financial risk to lending institutions. By analyzing customer demographics, credit information, repayment history, and payment behavior, this project aims to predict future default risk accurately and provide actionable insights.

---

## 📊 Dataset Information

**Dataset:** Default of Credit Card Clients Dataset (Taiwan)

**Records:** 30,000 customers

**Target Variable:**

* `0` → Non-Default
* `1` → Default

### Key Features

* LIMIT_BAL
* SEX
* EDUCATION
* MARRIAGE
* AGE
* PAY_0 to PAY_6 (Repayment Status)
* BILL_AMT1 to BILL_AMT6
* PAY_AMT1 to PAY_AMT6

---

## 🔍 Exploratory Data Analysis

Performed:

* Univariate Analysis
* Bivariate Analysis
* Correlation Analysis
* Class Distribution Analysis

### Key Findings

* Repayment history strongly influences default behavior.
* Customers with higher repayment delays are more likely to default.
* Higher credit utilization increases default risk.
* Larger credit limits reduce default probability.

---

## ⚙️ Feature Engineering

The following business-driven features were created:

### Avg_Delay

Average repayment delay across historical repayment records.

### Total_Payment

Total amount paid by the customer.

### Credit_Utilized

Ratio of average bill amount to available credit limit.

### Pay_Ratio

Ratio of total payment made to total billed amount.

---

## 🤖 Machine Learning Models

The following models were trained and evaluated:

### Logistic Regression

* ROC-AUC: 0.710

### Decision Tree (GridSearchCV Optimized)

* ROC-AUC: 0.756

### Random Forest (GridSearchCV Optimized)

* ROC-AUC: 0.774

### XGBoost (Final Model)

* ROC-AUC: **0.778**
* Accuracy: **77.0%**
* Precision: **48.41%**
* Recall: **60.96%**
* F1 Score: **53.97%**

---

## 📈 Model Comparison

| Model               |   Accuracy |  Precision |     Recall |   F1 Score |   ROC-AUC |
| ------------------- | ---------: | ---------: | ---------: | ---------: | --------: |
| Logistic Regression |     68.93% |     37.50% |     60.74% |     46.37% |     0.710 |
| Decision Tree       |     77.03% |     48.36% |     56.66% |     52.18% |     0.756 |
| Random Forest       |     64.63% |     35.81% |     75.65% |     48.61% |     0.774 |
| XGBoost             | **77.00%** | **48.41%** | **60.96%** | **53.97%** | **0.778** |

---

## 🔎 Model Explainability (SHAP)

SHAP (SHapley Additive exPlanations) was used to interpret model predictions.

### Most Influential Features

1. PAY_0
2. Avg_Delay
3. PAY_2
4. Total_Payment
5. LIMIT_BAL

SHAP analysis helped explain how individual features contributed to default risk predictions.

---

## 🖥️ Streamlit Application

An interactive Streamlit web application was developed to:

* Input customer details
* Predict default risk
* Classify customers as:

  * Low Risk
  * High Risk

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* XGBoost
* SHAP
* Streamlit
* Joblib

---

## 📂 Project Structure

```text
Credit_Default_Prediction/
│
├── app.py
├── Credit_Default_Prediction_Report.pdf
├── credit_default_model.pkl
├── scaler.pkl
├── requirements.txt
├── README.md
└── notebook.ipynb
```

## 🚀 Key Learnings

* Credit Risk Modeling
* Feature Engineering
* Class Imbalance Handling
* Hyperparameter Tuning using GridSearchCV
* XGBoost Implementation
* SHAP Explainability
* Model Deployment with Streamlit

---

## 📌 Conclusion

This project successfully developed an interpretable and accurate credit default prediction system. XGBoost achieved the best overall performance and was selected as the final model. The findings demonstrate that repayment behavior is the strongest indicator of future default risk, making the solution valuable for real-world credit risk assessment.

---

**Author:** Lakshay Gauniyal
**Project Type:** Machine Learning Classification Project
**Internship:** Innovexa Catalyst ML Internship
