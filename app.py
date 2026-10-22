from pathlib import Path
import joblib as jb
import pandas as pd
import streamlit as st

st.set_page_config(page_title="PremiumIQ", page_icon="🩺", layout="centered")

BASE = Path(__file__).parent
@st.cache_resource
def load_arti():
    col = jb.load(BASE /"columns.pkl")
    mod = jb.load(BASE /"insurance_model.pkl")
    return col ,mod

expected_columns ,model = load_arti()

st.title("🩺PremiumIQ")
st.caption("Estimate yearly medical insurance charges with a tuned Gradient Boosting model.")

st.markdown("provide the following details")
col1 ,col2 = st.columns(2)
with col1:
    age = st.slider("WHAT IS YOUR AGE",18,100,35)
    sex = st.selectbox("WHAT IS YOUR GENDER ",['MALE','FEMALE'])
    bmi = st.number_input("BMI",10.0,180.0,30.0,0.1)

with col2:
    children = st.number_input("HOW MANY CHILDREN DO YOU HAVE",0,20,1)
    smoker = st.selectbox("ARE YOU SMOKER",['YES','NO'])
    region = st.selectbox("WHAT IS YOUR REGION",["SOUTHWEST","SOUTHEAST","NORTHWEST","NORTHEAST"])

if st.button("predict"):
    raw_data={
        'age':age,
        'smoker':1 if smoker=="YES" else 0,
        'children':children,
        'region_southeast':1 if region == "SOUTHEAST" else 0,
        'is_male':1 if sex=="MALE" else 0,
        'bmi':bmi,
        'bmi_category_obese': 1 if bmi>=30 else 0
    }
    dt = pd.DataFrame([raw_data])
    for col in expected_columns:
        if col not in dt.columns:
            dt[col] = 0
    dt = dt[expected_columns]
    prediction = model.predict(dt)[0]
    st.success(f"Estimated yearly insurance charges: ${prediction:,.2f}")
    st.caption("Model: Gradient Boosting | Test R² ≈ 0.89 | Typical error ≈ ±$2,550")