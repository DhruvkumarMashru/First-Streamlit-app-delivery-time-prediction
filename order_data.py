import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# Generate Sample Data
data = {
    "Product_Category": np.random.choice(["Electronics", "Clothing", "Food", "Furniture"], 1000),
    "Customer_Location": np.random.choice(["Urban", "Suburban", "Rural"], 1000),
    "Shipping_Method": np.random.choice(["Standard", "Express", "Same-day"], 1000),
    "Delivery_Time": np.random.uniform(1, 10, 1000)
}
df = pd.DataFrame(data)

# Convert Categorical Variables to Numeric
df = pd.get_dummies(df, drop_first=True)

# Split Data
X = df.drop("Delivery_Time", axis=1)
y = df["Delivery_Time"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save Model
joblib.dump(model, "delivery_time_model.pkl")

# Save Sample Data
df.to_csv("order_data.csv", index=False)

print("Model training complete and data saved.")
