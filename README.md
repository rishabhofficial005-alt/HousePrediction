🏠 Bangalore House Price Prediction Web App
📌 Project Overview

This project is a Machine Learning–based web application that predicts house prices in Bangalore based on user inputs such as location, total square feet, number of bathrooms, and BHK.

The trained machine learning model is deployed using Streamlit, which is used purely as a frontend interface to interact with the model and display predictions.

🚀 Features

i-Predicts house prices in Lakhs and Crores

ii-Handles categorical (location) and numerical features

iii-User-friendly web interface built using Streamlit

iv-Uses a trained Random Forest Regression model

v-Clean and simple UI for easy interaction

🧠 Machine Learning Workflow

Data Collection
      ↓
Data Cleaning
      ↓
Feature Engineering
      ↓
Encoding Categorical Variables
      ↓
Train-Test Split
      ↓
Model Training (Random Forest)
      ↓
Model Evaluation
      ↓
Prediction via Streamlit Web App

🛠️ Tech Stack

Programming Language: Python

Data Analysis: Pandas, NumPy

Machine Learning: Scikit-learn

Model Used: Random Forest Regressor

Frontend / UI: Streamlit

Model Serialization: Joblib

📁 Project Structure
Banglore-House-Prediction/
│
├── app.py                       # Streamlit frontend
├── data/
│   └── Cleaned_data.csv         # Cleaned dataset
├── notebooks/
│   └── HousePrediction.ipynb    # Model training
├── assets/
│   └── background.jpg           # UI image

Note: The trained model file (.pkl) is not included in the repository due to GitHub file size limitations.


🎯 Future Improvements

Location dropdown instead of text input

Improved feature engineering

More advanced regression models

Better UI/UX design

Cloud deployment

🙌 Key Learnings

End-to-end machine learning project development

Handling categorical and numerical features

Training and evaluating regression models

Deploying ML models using Streamlit

Designing simple and effective ML web interface
