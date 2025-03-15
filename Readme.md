# Delivery Time Prediction App

This project is a web-based application that predicts the estimated delivery time for an order based on product category, customer location, and shipping method.

## Features
- User-friendly Streamlit interface
- Machine Learning model for delivery time prediction
- Data visualization of delivery times

## Setup Instructions

### 1. Install Dependencies
```
pip install -r requirements.txt
```

### 2. Train the Model (Optional)
Run the following command to generate sample data and train the model:
```
python order_data.py
```

### 3. Run the Streamlit App
```
streamlit run app.py
```

The app will open in your browser, allowing you to enter order details and get a delivery time prediction.

## File Structure
- `app.py` – Streamlit application
- `order_data.py` – Script to generate sample data and train the model
- `order_data.csv` – Sample dataset
- `delivery_time_model.pkl` – Trained model
- `requirements.txt` – Required dependencies
- `README.md` – Project instructions
