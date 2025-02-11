import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model (Using GradientBoostingRegressor)
model = joblib.load("car_price_model.pkl")

# Load processed data to get unique values for dropdowns
df = pd.read_csv("Dataset/processed_car_data.csv")

st.title("🚗 Car Price Prediction App")
st.write("Enter car details to predict its selling price.")

# User input fields
year = st.number_input("Manufacturing Year", min_value=2000, max_value=2025, step=1)
km_driven = st.number_input("Kilometers Driven", min_value=0, step=1000)
mileage = st.number_input("Mileage (kmpl)", min_value=5.0, max_value=40.0, step=0.1)
engine = st.number_input("Engine Capacity (CC)", min_value=500, max_value=5000, step=10)
max_power = st.number_input("Max Power (bhp)", min_value=10.0, max_value=500.0, step=1.0)

fuel = st.selectbox("Fuel Type", df["fuel"].unique())
seller_type = st.selectbox("Seller Type", df["seller_type"].unique())
transmission = st.selectbox("Transmission", df["transmission"].unique())
owner = st.selectbox("Owner Type", df["owner"].unique())

if st.button("Predict Price"):
    # Create input DataFrame
    input_data = pd.DataFrame([[year, np.log1p(km_driven), mileage, engine, max_power, fuel, seller_type, transmission, owner]],
                              columns=["year", "km_driven", "mileage", "engine", "max_power", "fuel", "seller_type", "transmission", "owner"])
    
    # Predict price
    prediction = model.predict(input_data)[0]
    predicted_price = np.expm1(prediction)
    
    st.success(f"Estimated Selling Price: ₹{round(predicted_price, 2)}")