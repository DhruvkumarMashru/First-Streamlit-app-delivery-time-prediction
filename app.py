import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os
import subprocess

try:
    import joblib
except ImportError:
    subprocess.run(["pip", "install", "joblib"])
    import joblib


# Load the trained model
model = joblib.load("delivery_time_model.pkl")

# Get expected feature names from the model
expected_features = model.feature_names_in_

def predict_delivery_time(product_category, customer_location, shipping_method):
    # Create an empty DataFrame with correct feature names
    input_data = pd.DataFrame(columns=expected_features)
    input_data.loc[0] = 0  # Initialize with zeros
    
    # Set the correct feature flags (One-Hot Encoding)
    if f"Product_Category_{product_category}" in input_data.columns:
        input_data[f"Product_Category_{product_category}"] = 1
    if f"Customer_Location_{customer_location}" in input_data.columns:
        input_data[f"Customer_Location_{customer_location}"] = 1
    if f"Shipping_Method_{shipping_method}" in input_data.columns:
        input_data[f"Shipping_Method_{shipping_method}"] = 1
    
    # Predict delivery time
    predicted_time = model.predict(input_data)[0]
    return predicted_time

# Streamlit UI
st.title("📦 Order Delivery Time Prediction")

st.sidebar.header("Input Order Details")
product_category = st.sidebar.selectbox("Product Category", ["Clothing", "Electronics", "Food", "Furniture"])
customer_location = st.sidebar.selectbox("Customer Location", ["Urban", "Suburban", "Rural"])
shipping_method = st.sidebar.selectbox("Shipping Method", ["Standard", "Express"])

if st.sidebar.button("Predict Delivery Time"):
    try:
        prediction = predict_delivery_time(product_category, customer_location, shipping_method)
        st.success(f"Estimated Delivery Time: {prediction:.2f} days")

        # Improved Visualization: Horizontal Bar Chart
        fig, ax = plt.subplots()
        ax.barh(["Predicted Delivery Time"], [prediction], color='skyblue', height=0.4)
        ax.set_xlabel("Days")
        ax.set_title("Predicted Delivery Time Visualization")

        # Adding the value label on the bar
        for index, value in enumerate([prediction]):
            ax.text(value, index, f"{value:.2f} days", va='center', fontsize=12, fontweight='bold', color='black')

        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error: {e}")
