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

sex_mapping = {1:"Male",2:"Female"}
SEX = st.selectbox("Sex", [1, 2], format_func= lambda x: sex_mapping[x])

education_mapping = {1:"Graduate School",2:"University",3:"High School",4:"Others",5:"Unknown"}
EDUCATION = st.selectbox("Education", [1, 2, 3, 4, 5], format_func=lambda x:education_mapping.get(x))

marrige_mapping = {1:"Married",2:"Single",3:"Others"}
MARRIAGE = st.selectbox("Marriage", [1, 2, 3], format_func=lambda x: marrige_mapping.get(x))

repayment_map = {
    -2:"-2: No consumption / no bill issued",
    -1:"-1: Paid on time",
    0:"0: Used credit but paid without Delay",
    1:"1: Payment delay for 1 month",
    2:"2: Payment delay for 2 months",
    3:"3: Payment delay for 3 months",
    4:"4: Payment delay for 4 months",
    5:"5: Payment delay for 5 months",
    6:"6: Payment delay for 6 months",
    7:"7: Payment delay for 7 months",
    8:"8: Payment delay for 8 months",
    9:"9: Payment delay for 9 months"
}

repayment_options = list(range(-2,10))
PAY_0 = st.selectbox("Current Repayment Status",options=repayment_options,format_func=lambda x: repayment_map[x])
PAY_2 = st.selectbox("Previous Repayment Status",options=repayment_options,format_func=lambda x: repayment_map[x])

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
