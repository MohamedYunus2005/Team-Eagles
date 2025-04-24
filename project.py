import pandas as pd
import numpy as np

data = {
    "Square_Footage": [1500, 2000, 2500, 3000, 3500],
    "Bedrooms": [3, 4, 3, 5, 4],
    "Location_Score": [7, 8, 7, 9, 8],
    "Price": [300000, 400000, 350000, 500000, 450000]
}
df = pd.DataFrame(data)

X = df[["Square_Footage", "Bedrooms", "Location_Score"]]
X = np.c_[np.ones(X.shape[0]), X]
y = df["Price"].values

split_index = int(0.8 * len(df))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

theta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train

y_pred = X_test @ theta

mse = np.mean((y_test - y_pred)**2)
r2 = 1 - np.sum((y_test - y_pred)**2) / np.sum((y_test - np.mean(y_test))**2)

print("Mean Squared Error:", mse)
print("R² Score:", r2)

new_data = np.array([[1, 1800, 3, 8]])
predicted_price = new_data @ theta
print("Predicted Price:", predicted_price[0])
