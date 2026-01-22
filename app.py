import os
import joblib
import pandas as pd
import streamlit as st
import base64
import urllib.request

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Bangalore House Price Prediction",
    layout="centered"
)

# -------------------------------------------------
# BACKGROUND FUNCTION
# -------------------------------------------------
def set_bg(image_path):
    if not os.path.exists(image_path):
        st.warning("Background image not found.")
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

        .card {{
            background-color: rgba(255, 255, 255, 0.97);
            padding: 30px;
            border-radius: 15px;
            max-width: 700px;
            margin: auto;
            box-shadow: 0px 10px 25px rgba(0,0,0,0.25);
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# -------------------------------------------------
# APPLY BACKGROUND
# -------------------------------------------------
set_bg("background.jpg")

# -------------------------------------------------
# MODEL DOWNLOAD & LOAD (FINAL, STABLE)
# -------------------------------------------------
MODEL_ID = "10VD7Petb9b4CLoo0jfgJA5oPR_r4p4Tt"   # NEW retrained model
MODEL_PATH = "final_house_price_model.pkl"

def download_model():
    url = f"https://drive.google.com/uc?export=download&id={MODEL_ID}"
    with st.spinner("📥 Downloading ML model (first run only)..."):
        urllib.request.urlretrieve(url, MODEL_PATH)

def load_model():
    try:
        if not os.path.exists(MODEL_PATH):
            download_model()
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.error("❌ Failed to load ML model.")
        st.exception(e)
        st.stop()

model = load_model()

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

st.markdown("<div class='card'>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    location = st.selectbox("📍 Select Location", locations)
    bath = st.number_input("🛁 Bathrooms", min_value=1, step=1)

with col2:
    bhk = st.number_input("🛏️ BHK", min_value=1, step=1)
    sqft = st.number_input("📐 Total SQFT", min_value=300)

# -------------------------------------------------
# PREDICTION
# -------------------------------------------------
if st.button("🔮 Predict Price"):
    input_df = pd.DataFrame({
        "location": [location],
        "total_sqft": [sqft],
        "bath": [bath],
        "bhk": [bhk]
    })

    prediction = model.predict(input_df)[0]
    st.success(f"💰 Estimated Price: ₹ {round(prediction, 2)} Lakhs")

st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.markdown(
    "<br><hr><p style='text-align:center;'>Deployed on Streamlit Cloud</p>",
    unsafe_allow_html=True
)
