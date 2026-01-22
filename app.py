import os
import joblib
import pandas as pd
import streamlit as st
import base64
import gdown
import traceback

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Bangalore House Price Prediction",
    layout="centered"
)

# -----------------------------
# BACKGROUND FUNCTION
# -----------------------------
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

# -----------------------------
# APPLY BACKGROUND
# -----------------------------
set_bg("background.jpg")

# -----------------------------
# LOAD MODEL (DEMO MODE SAFE)
# -----------------------------
# -------------------------
# MODEL DOWNLOAD & LOAD (robust)
# -------------------------
import gdown
import joblib
import traceback

MODEL_ID = "1bzyAflVPjeFnj5Z3BKoG2nLZQkC_x4nn"
MODEL_PATH = "final_house_price_model.pkl"

def load_model_safe(model_id=MODEL_ID, model_path=MODEL_PATH):
    # 1) If file already exists in app folder, try load it first
    if os.path.exists(model_path):
        try:
            return joblib.load(model_path)
        except Exception as e:
            st.error("Existing model file found but failed to load. Will attempt re-download.")
            st.exception(e)

    # 2) Download from Google Drive using gdown (handles large files)
    try:
        url = f"https://drive.google.com/uc?id={model_id}"
        with st.spinner("📥 Downloading model from Google Drive (first run only)..."):
            gdown.download(url, model_path, quiet=False)
    except Exception as e:
        st.error("Failed to download model from Google Drive.")
        st.exception(e)
        return None

    # 3) Try to load the downloaded model
    try:
        return joblib.load(model_path)
    except Exception as e:
        st.error("Downloaded file could not be loaded as a model.")
        st.exception(e)
        # Helpful hint for the user
        st.info("If you see an AttributeError related to sklearn internals, the model was "
                "pickled with a different scikit-learn version than the runtime. "
                "Run `python -c \"import sklearn; print(sklearn.__version__)\"` locally and pin that version in requirements.txt.")
        return None

# call once
model = load_model_safe()


# -----------------------------
# LOAD DATA (SAFE)
# -----------------------------
DATA_PATH = "Cleaned_data.csv"

if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
    locations = sorted(df["location"].unique())
else:
    st.error("Dataset not found.")
    st.stop()

# -----------------------------
# UI
# -----------------------------
st.markdown(
    "<h1 style='text-align:center;color:white;'>Welcome to Bangalore House Prediction App</h1>",
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:
    location = st.selectbox("Select Location", locations)
    bath = st.number_input("Bathrooms", min_value=1, step=1)

with col2:
    bhk = st.number_input("BHK", min_value=1, step=1)
    sqft = st.number_input("Total SQFT", min_value=1)

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("Predict Price"):
    input_df = pd.DataFrame({
        "location": [location],
        "total_sqft": [sqft],
        "bath": [bath],
        "bhk": [bhk]
    })

    if model is None:
        st.info("🔍 Prediction disabled in demo mode (model not deployed).")
    else:
        prediction = model.predict(input_df)[0]
        st.success(f"Estimated Price: ₹ {round(prediction, 2)} Lakhs")
