import sys
from types import ModuleType

if "pkg_resources" not in sys.modules:
    fake_pkg = ModuleType("pkg_resources")

    class DistributionNotFound(Exception):
        pass

    fake_pkg.DistributionNotFound = DistributionNotFound

    class FakeDistribution:
        def __init__(self):
            self.version = "1.0.0"

    fake_pkg.get_distribution = lambda *args, **kwargs: FakeDistribution()
    fake_pkg.resource_filename = lambda *args, **kwargs: ""
    sys.modules["pkg_resources"] = fake_pkg

import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load("credit_default_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Credit Card Default Predictor")

st.write("Enter customer details")

LIMIT_BAL = st.number_input("Credit Limit")
AGE = st.number_input("Age")

SEX = st.selectbox("Sex", [1, 2])

EDUCATION = st.selectbox("Education", [1, 2, 3, 4])

MARRIAGE = st.selectbox("Marriage", [1, 2, 3])

PAY_0 = st.number_input("Current Repayment Status")
PAY_2 = st.number_input("Previous Repayment Status")

Avg_Delay = st.number_input("Average Delay")
Total_Payment = st.number_input("Total Payment")
Credit_Utilized = st.number_input("Credit Utilisation")
Pay_Ratio = st.number_input("Payment Ratio")

if st.button("Predict"):

    data = pd.DataFrame(
        [
            [
                LIMIT_BAL,
                SEX,
                EDUCATION,
                MARRIAGE,
                AGE,
                PAY_0,
                PAY_2,
                Avg_Delay,
                Total_Payment,
                Credit_Utilized,
                Pay_Ratio,
            ]
        ],
        columns=[
            "LIMIT_BAL",
            "SEX",
            "EDUCATION",
            "MARRIAGE",
            "AGE",
            "PAY_0",
            "PAY_2",
            "Avg_Delay",
            "Total_Payment",
            "Credit_Utilized",
            "Pay_Ratio",
        ],
    )

    num_cols = [
        "LIMIT_BAL",
        "AGE",
        "Avg_Delay",
        "Total_Payment",
        "Credit_Utilized",
        "Pay_Ratio",
    ]

    data[num_cols] = scaler.transform(data[num_cols])

    prediction = model.predict(data)[0]

    if prediction == 1:
        st.error("High Risk of Default")
    else:
        st.success("Low Risk of Default")
