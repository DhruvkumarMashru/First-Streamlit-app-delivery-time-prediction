import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained model
model = joblib.load("delivery_time_model.pkl")

# Streamlit UI Setup
st.set_page_config(page_title="Delivery Time Prediction", layout="wide")
st.title("📦 Order Delivery Time Prediction")
st.write("Enter order details to predict the estimated delivery time.")

# Sidebar for Inputs
st.sidebar.header("Input Order Details")
product_category = st.sidebar.selectbox("Product Category", ["Electronics", "Clothing", "Food", "Furniture"])
customer_location = st.sidebar.selectbox("Customer Location", ["Urban", "Suburban", "Rural"])
shipping_method = st.sidebar.selectbox("Shipping Method", ["Standard", "Express", "Same-day"])

# Create a DataFrame for Prediction
input_data = pd.DataFrame({
    "Product_Category": [product_category],
    "Customer_Location": [customer_location],
    "Shipping_Method": [shipping_method]
})

# Make Prediction
if st.sidebar.button("Predict Delivery Time"):
    predicted_time = model.predict(input_data)[0]
    st.success(f"🕒 Expected Delivery Time: {predicted_time:.2f} days")

# Load Sample Data for Visualization
data = pd.read_csv("order_data.csv")

# Visualization: Distribution of Delivery Times
st.subheader("📊 Delivery Time Distribution")
fig, ax = plt.subplots()
sns.histplot(data["Delivery_Time"], bins=20, kde=True, ax=ax)
ax.set_xlabel("Delivery Time (Days)")
st.pyplot(fig)
