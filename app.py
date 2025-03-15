import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Load the trained model safely
model_path = "model.pkl"

if not model_path or not isinstance(model_path, str):
    st.error("Model file path is invalid!")
    st.stop()

try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    expected_features = model.feature_names_in_
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Prediction function
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
st.set_page_config(page_title="Delivery Time Predictor", page_icon="🚚", layout="wide")

st.title("📦 Order Delivery Time Prediction")
st.markdown("### Enter order details to get an estimated delivery time.")

# Sidebar Inputs
with st.sidebar:
    st.header("📋 Input Order Details")
    product_category = st.selectbox("Select Product Category", ["Clothing", "Electronics", "Food", "Furniture"])
    customer_location = st.selectbox("Select Customer Location", ["Urban", "Suburban", "Rural"])
    shipping_method = st.selectbox("Select Shipping Method", ["Standard", "Express"])
    predict_button = st.button("🚀 Predict Delivery Time")

# Main Content
if predict_button:
    try:
        prediction = predict_delivery_time(product_category, customer_location, shipping_method)
        st.success(f"🕒 Estimated Delivery Time: **{prediction:.2f} days**")

        # Visualization
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.barh(["Predicted Delivery Time"], [prediction], color='skyblue', height=0.4)
        ax.set_xlabel("Days")
        ax.set_title("Predicted Delivery Time Visualization")

        # Add text on the bar
        ax.text(prediction, 0, f"{prediction:.2f} days", va='center', fontsize=12, fontweight='bold')

        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error: {e}")
