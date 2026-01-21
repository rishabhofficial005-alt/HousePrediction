import os
import joblib
import pandas as pd
import streamlit as st
import base64
import gdown

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Bangalore House Price Prediction",
    layout="centered"
)

# -------------------------------------------------
# BACKGROUND
# -------------------------------------------------
def set_bg(image_path):
    if not os.path.exists(image_path):
        return
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-repeat: no-repeat;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg("background.jpg")

# -------------------------------------------------
# MODEL DOWNLOAD & LOAD (NO CACHE – STABLE)
# -------------------------------------------------
MODEL_ID = "1bzyAflVPjeFnj5Z3BKoG2nLZQkC_x4nn"
MODEL_PATH = "final_house_price_model.pkl"

try:
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📥 Downloading ML model (first run only)..."):
            gdown.download(
                f"https://drive.google.com/uc?id={MODEL_ID}",
                MODEL_PATH,
                quiet=False
            )

    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error("❌ Failed to load ML model.")
    st.exception(e)
    st.stop()

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
DATA_PATH = "Cleaned_data.csv"

if not os.path.exists(DATA_PATH):
    st.error("Dataset not found.")
    st.stop()

df = pd.read_csv(DATA_PATH)
locations = sorted(df["location"].unique())

# -------------------------------------------------
# UI
# -------------------------------------------------
st.markdown(
    "<h1 style='text-align:center;color:white;'>🏠 Bangalore House Price Prediction</h1>",
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:
    location = st.selectbox("📍 Location", locations)
    bath = st.number_input("Bathrooms", min_value=1, step=1)

with col2:
    bhk = st.number_input("BHK", min_value=1, step=1)
    sqft = st.number_input("Total SQFT", min_value=300)

# -------------------------------------------------
# PREDICTION
# -------------------------------------------------
if st.button("Predict Price"):
    input_df = pd.DataFrame({
        "location": [location],
        "total_sqft": [sqft],
        "bath": [bath],
        "bhk": [bhk]
    })

    prediction = model.predict(input_df)[0]
    st.success(f"💰 Estimated Price: ₹ {prediction:.2f} Lakhs")

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.markdown(
    "<hr><p style='text-align:center;'>Deployed on Streamlit Cloud</p>",
    unsafe_allow_html=True
)
