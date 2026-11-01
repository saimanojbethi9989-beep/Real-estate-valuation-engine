import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Configure web page
st.set_page_config(
    page_title="Real Estate Valuation Engine", page_icon="🏡", layout="centered"
)


# Load the saved model
@st.cache_resource
def load_model():
    return joblib.load("trained_model.joblib")


try:
    model = load_model()
except Exception:
    st.error("Model file not found! Please run 'python train.py' first.")
    st.stop()

# Header
st.title("🏡 Real Estate Valuation Engine")
st.write(
    "Adjust the neighborhood and home specifications below to get an instant AI market estimate."
)
st.divider()

# Input layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Property Specs")
    income = st.slider("Median Local Income ($k/year)", 15.0, 150.0, 65.0, 5.0)
    house_age = st.slider("House Age (Years)", 1, 60, 20)
    ave_rooms = st.slider("Average Total Rooms", 2.0, 12.0, 5.5, 0.5)
    ave_bedrms = st.slider("Average Bedrooms", 0.5, 5.0, 1.2, 0.1)

with col2:
    st.subheader("Location Factors")
    population = st.number_input("Neighborhood Population", 100, 10000, 1200)
    ave_occup = st.number_input("Average Household Size", 1.0, 8.0, 3.0)
    latitude = st.slider("Latitude", 32.5, 42.0, 37.8, 0.1)
    longitude = st.slider("Longitude", -124.5, -114.0, -122.2, 0.1)

# Dynamically calculate the engineered features
safe_rooms = 1e-5 if ave_rooms == 0 else ave_rooms
bedrooms_ratio = ave_bedrms / safe_rooms
density_per_room = population / (safe_rooms + 1)

# Package into a DataFrame that matches the training format
input_record = pd.DataFrame(
    [
        {
            "MedInc": income / 10.0,
            "HouseAge": house_age,
            "AveRooms": ave_rooms,
            "AveBedrms": ave_bedrms,
            "Population": population,
            "AveOccup": ave_occup,
            "Latitude": latitude,
            "Longitude": longitude,
            "BedroomsRatio": bedrooms_ratio,
            "DensityPerRoom": density_per_room,
        }
    ]
)

st.divider()

# Predict Button
if st.button("Calculate Market Valuation", type="primary", use_container_width=True):
    prediction = model.predict(input_record)[0]
    final_price = max(20000, prediction)

    st.success("Valuation Complete!")
    st.metric(
        label="Estimated Property Market Value", value=f"${final_price:,.2f}"
    )
    st.caption(f"Coordinates: {latitude}, {longitude} | Processed via XGBoost")