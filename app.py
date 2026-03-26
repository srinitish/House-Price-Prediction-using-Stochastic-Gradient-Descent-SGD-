import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="centered")

# ---------- STYLE ----------
st.markdown("""
<style>

.stApp {
    background: linear-gradient(120deg,#74ebd5,#ACB6E5);
}

.card {
    background:white;
    padding:30px;
    border-radius:12px;
    box-shadow:0px 6px 20px rgba(0,0,0,0.15);
}

.title {
    text-align:center;
    font-size:36px;
    font-weight:bold;
    color:#2C3E50;
}

.result {
    font-size:32px;
    text-align:center;
    font-weight:bold;
    color:#1B5E20;
    background:#E8F5E9;
    padding:20px;
    border-radius:10px;
}

</style>
""", unsafe_allow_html=True)

# ---------- LOAD MODEL ----------
model = joblib.load("model.pkl")
scaler_x = joblib.load("scaler_x.pkl")
scaler_y = joblib.load("scaler_y.pkl")

# ---------- CARD ----------


st.markdown('<p class="title">🏠 House Price Prediction</p>', unsafe_allow_html=True)

st.write("Enter house details to estimate the price.")

st.divider()

col1, col2 = st.columns(2)

with col1:
    area = st.slider("Area (sqft)", 500, 5000, 1200)

with col2:
    bedrooms = st.slider("Bedrooms", 1, 6, 2)

st.divider()

if st.button("Predict Price 💰", use_container_width=True):

    input_data = np.array([[area, bedrooms]])
    scaled = scaler_x.transform(input_data)

    prediction_scaled = model.predict(scaled)
    prediction = scaler_y.inverse_transform(prediction_scaled)

    price = prediction[0][0]

    st.markdown(
        f'<div class="result">Estimated Price: ₹ {price:,.2f} Lakhs</div>',
        unsafe_allow_html=True
    )

st.markdown("</div>", unsafe_allow_html=True)