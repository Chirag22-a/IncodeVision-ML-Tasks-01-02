"""Streamlit form for trying the saved customer churn model."""

from pathlib import Path

import pandas as pd
import streamlit as st
from joblib import load


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "reports" / "churn_pipeline.joblib"

st.set_page_config(page_title="Customer Churn Risk", page_icon="📊")
st.title("Customer Churn Risk Demo")
st.caption(
    "Enter a sample customer profile. This educational model estimates risk; "
    "it is not a guarantee or a production decision."
)

if not MODEL_PATH.exists():
    st.warning("The model has not been trained yet. Run python main.py first.")
    st.stop()

model = load(MODEL_PATH)
with st.form("customer_profile"):
    left, right = st.columns(2)

    with left:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox(
            "Senior citizen",
            [0, 1],
            format_func=lambda value: "Yes" if value == 1 else "No",
        )
        partner = st.selectbox("Has a partner?", ["Yes", "No"])
        dependents = st.selectbox("Has dependents?", ["Yes", "No"])
        tenure = st.number_input(
            "Tenure in months",
            min_value=0,
            max_value=72,
            value=12,
            step=1,
        )
        phone_service = st.selectbox("Phone service", ["Yes", "No"])
        if phone_service == "No":
            multiple_lines = "No phone service"
            st.selectbox(
                "Multiple lines",
                ["No phone service"],
                disabled=True,
            )
        else:
            multiple_lines = st.selectbox("Multiple lines", ["No", "Yes"])

        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"],
        )
        paperless_billing = st.selectbox("Paperless billing", ["Yes", "No"])

    with right:
        internet_service = st.selectbox(
            "Internet service",
            ["DSL", "Fiber optic", "No"],
        )
        if internet_service == "No":
            no_internet = ["No internet service"]
            online_security = st.selectbox(
                "Online security", no_internet, disabled=True
            )
            online_backup = st.selectbox(
                "Online backup", no_internet, disabled=True
            )
            device_protection = st.selectbox(
                "Device protection", no_internet, disabled=True
            )
            tech_support = st.selectbox(
                "Tech support", no_internet, disabled=True
            )
            streaming_tv = st.selectbox(
                "Streaming TV", no_internet, disabled=True
            )
            streaming_movies = st.selectbox(
                "Streaming movies", no_internet, disabled=True
            )
        else:
            online_security = st.selectbox("Online security", ["No", "Yes"])
            online_backup = st.selectbox("Online backup", ["No", "Yes"])
            device_protection = st.selectbox("Device protection", ["No", "Yes"])
            tech_support = st.selectbox("Tech support", ["No", "Yes"])
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes"])
            streaming_movies = st.selectbox("Streaming movies", ["No", "Yes"])

        payment_method = st.selectbox(
            "Payment method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )
        monthly_charges = st.number_input(
            "Monthly charges",
            min_value=0.0,
            max_value=500.0,
            value=70.0,
            step=1.0,
        )
        total_charges = st.number_input(
            "Total charges so far",
            min_value=0.0,
            max_value=100000.0,
            value=840.0,
            step=10.0,
        )

    submitted = st.form_submit_button("Estimate churn risk")

if submitted:
    profile = pd.DataFrame(
        [
            {
                "gender": gender,
                "SeniorCitizen": senior_citizen,
                "Partner": partner,
                "Dependents": dependents,
                "tenure": tenure,
                "PhoneService": phone_service,
                "MultipleLines": multiple_lines,
                "InternetService": internet_service,
                "OnlineSecurity": online_security,
                "OnlineBackup": online_backup,
                "DeviceProtection": device_protection,
                "TechSupport": tech_support,
                "StreamingTV": streaming_tv,
                "StreamingMovies": streaming_movies,
                "Contract": contract,
                "PaperlessBilling": paperless_billing,
                "PaymentMethod": payment_method,
                "MonthlyCharges": monthly_charges,
                "TotalCharges": total_charges,
            }
        ]
    )
    churn_probability = float(model.predict_proba(profile)[0, 1])
    predicted_churn = model.predict(profile)[0] == 1

    st.metric("Estimated churn probability", f"{churn_probability:.1%}")
    st.progress(churn_probability)
    if predicted_churn:
        st.warning("The model classifies this profile as higher churn risk.")
    else:
        st.success("The model classifies this profile as lower churn risk.")
    st.caption(
        "This estimate comes from a small sample dataset and should not be used "
        "to make real customer decisions."
    )
