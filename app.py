import joblib 
import pandas as pd
import streamlit as st 
import base64

st.set_page_config(
    page_title="Bangalore House Price Prediction",
    layout="centered"
)

def set_bg(image_path):
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
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# CALL THE FUNCTION
set_bg("background.jpg")

model = joblib.load(
    r"C:\Users\KIIT0001\Desktop\Banglore House Prediciton\final_house_price_model.pkl"
)

df = pd.read_csv(
    r"C:\Users\KIIT0001\Desktop\Banglore House Prediciton\Cleaned_data.csv"
)

locations = sorted(df['location'].unique())

st.markdown(
    "<h1 style='text-align:center;color:white;'>Welcome to Banglore House Prediction App</h1>",
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:
    location = st.selectbox("Select Location", locations)
    bath = st.number_input("Bathrooms", min_value=1, step=1)

with col2:
    bhk = st.number_input("BHK", min_value=1, step=1)
    sqft = st.number_input("Total SQFT", min_value=1)

if st.button("Predict Price"):
    input_df = pd.DataFrame({
        "location": [location],
        "total_sqft": [sqft],
        "bath": [bath],
        "bhk": [bhk]
    })

    prediction = model.predict(input_df)[0]
    st.success(f"Estimated Price: ₹ {round(prediction, 2)} Lakhs")
