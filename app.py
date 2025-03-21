import streamlit as st
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

st.title("🌾 Crop Recommendation System")
st.markdown("Enter the soil and weather parameters below to get a crop recommendation:")

# Input features
N = st.number_input("Nitrogen content (N)")
P = st.number_input("Phosphorus content (P)")
K = st.number_input("Potassium content (K)")
temperature = st.number_input("Temperature (°C)")
humidity = st.number_input("Humidity (%)")
ph = st.number_input("pH level")
rainfall = st.number_input("Rainfall (mm)")

# Predict
if st.button("Predict Crop"):
    input_features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = model.predict(input_features)
    st.success(f"✅ Recommended Crop: **{prediction[0]}**")