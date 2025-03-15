import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
os.system(f"{sys.executable} -m pip install --user joblib")

import joblib  # Now it should import successfully
import sys
print("Python executable path:", sys.executable)
print("Python version:", sys.version)

import altair as alt  
import subprocess
subprocess.run(["pip", "list"])  # This will print all installed packages



# ✅ Load the Model
model_path = "delivery_time_model.pkl"

if not os.path.exists(model_path):
    st.error("🚨 Model file 'delivery_time_model.pkl' not found! Make sure it's in the correct directory.")
    st.stop()

model = joblib.load(model_path)

# ✅ Streamlit Sidebar Inputs
st.sidebar.title("🔧 Order Details")
product_category = st.sidebar.selectbox("Select Product Category", ["Electronics", "Clothing", "Furniture", "Books", "Others"])
customer_location = st.sidebar.selectbox("Select Customer Location", ["Urban", "Suburban", "Rural"])
shipping_method = st.sidebar.selectbox("Select Shipping Method", ["Standard", "Express", "Same-Day"])
order_quantity = st.sidebar.number_input("Enter Order Quantity", min_value=1, step=1)

# ✅ Use Day Names Instead of Numbers
days_dict = {
    "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
    "Friday": 4, "Saturday": 5, "Sunday": 6
}
order_day = st.sidebar.selectbox("Purchased Day", list(days_dict.keys()))  # Dropdown with full names
order_hour = st.sidebar.slider("Purchased Hour (0-23)", min_value=0, max_value=23)  # Slider

# 🔍 **New Feature (Possible Missing Feature)**
distance = st.sidebar.selectbox("Shipping Distance (km)", ["Short", "Medium", "Long"])

# Convert Inputs to Model-Compatible Format
feature_dict = {
    "Electronics": 0, "Clothing": 1, "Furniture": 2, "Books": 3, "Others": 4,
    "Urban": 0, "Suburban": 1, "Rural": 2,
    "Standard": 0, "Express": 1, "Same-Day": 2,
    "Short": 0, "Medium": 1, "Long": 2  
}

# ✅ Ensure Feature Count Matches Model
input_features = np.array([
    feature_dict[product_category], 
    feature_dict[customer_location], 
    feature_dict[shipping_method],
    order_quantity, 
    days_dict[order_day],  # Use full day name instead of number
    order_hour,
    feature_dict[distance]  
]).reshape(1, -1)  

# 🔍 Predict Delivery Time
if st.sidebar.button("Predict Delivery Time"):
    predicted_time = model.predict(input_features)[0]
    
    # ✅ Display the estimated delivery time
    st.subheader(f"📅 Estimated Delivery Time: **{predicted_time:.2f} days**")

    # ✅ 📊 Improved Graph Visualization 
    chart_data = pd.DataFrame({
        "Delivery Estimate": ["Predicted Delivery Time"],
        "Days": [predicted_time]
    })

    chart = alt.Chart(chart_data).mark_bar(size=40).encode(
        x=alt.X("Days:Q", title="Estimated Days", scale=alt.Scale(domain=(0, predicted_time + 2))),
        y=alt.Y("Delivery Estimate:N", title=""),
        color=alt.value("#4C72B0"),  # Professional blue color
        tooltip=["Days"]
    ).properties(
        title="📊 Estimated Delivery Time",
        width=600,
        height=200
    )

    st.altair_chart(chart, use_container_width=True)
# Footer
st.markdown("Here is the Estimation of the order delivery in your Area.......")
