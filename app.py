import streamlit as st
import pandas as pd
import numpy as np
import pickle
import altair as alt  # Altair is natively supported by Streamlit

# Load the trained model
with open("delivery_time_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Get expected feature names from the model
expected_features = model.feature_names_in_

def predict_delivery_time(product_category, customer_location, shipping_method):
    input_data = pd.DataFrame(columns=expected_features)
    input_data.loc[0] = 0  # Initialize with zeros

    if f"Product_Category_{product_category}" in input_data.columns:
        input_data[f"Product_Category_{product_category}"] = 1
    if f"Customer_Location_{customer_location}" in input_data.columns:
        input_data[f"Customer_Location_{customer_location}"] = 1
    if f"Shipping_Method_{shipping_method}" in input_data.columns:
        input_data[f"Shipping_Method_{shipping_method}"] = 1

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

        # 📊 Visualization using Altair
        chart_data = pd.DataFrame({"Category": ["Predicted Delivery Time"], "Days": [prediction]})
        chart = alt.Chart(chart_data).mark_bar().encode(
            x="Days:Q",
            y="Category:N",
            color=alt.value("skyblue"),
            tooltip=["Days"]
        ).properties(title="Predicted Delivery Time Visualization")

        st.altair_chart(chart, use_container_width=True)

    except Exception as e:
        st.error(f"Error: {e}")
