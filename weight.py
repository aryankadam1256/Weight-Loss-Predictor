import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Generating synthetic dataset
np.random.seed(42)
size = 500

age = np.random.randint(18, 60, size)
weight = np.random.randint(50, 120, size)
height = np.random.randint(150, 200, size) / 100  # Convert to meters
caloric_intake = np.random.randint(1500, 3000, size)
time_duration = np.random.randint(30, 120, size)  # Minutes per day

gender = np.random.choice(["Male", "Female"], size)
activity_level = np.random.choice(["Low", "Moderate", "High"], size)

# True weight loss function (simulated)
true_w = [0.1091, -0.1148, -0.0144, -0.0024, 0.0058, -0.1537, 0.2284, 0.0899]
true_b = 15.1325

# Encoding categorical features
encoder = OneHotEncoder(drop='first', sparse_output=False)
categorical_data = encoder.fit_transform(np.column_stack((gender, activity_level)))

# Feature matrix
X = np.column_stack((age, weight, height, caloric_intake, time_duration, categorical_data))

# Weight loss target variable
noise = np.random.normal(0, 2, size)  # Adding some noise
y = (
    true_w[0] * age +
    true_w[1] * weight +
    true_w[2] * height +
    true_w[3] * caloric_intake +
    true_w[4] * time_duration +
    true_w[5] * categorical_data[:, 0] +
    true_w[6] * categorical_data[:, 1] +
    true_w[7] * categorical_data[:, 2] +
    true_b + noise
)

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Applying Ridge Regression
ridge_model = Ridge(alpha=0.1)
ridge_model.fit(X_train, y_train)

y_pred = ridge_model.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")

# Extracting learned parameters
learned_w = ridge_model.coef_
learned_b = ridge_model.intercept_
print("Learned Weights (W):", learned_w)
print("Learned Bias (b):", learned_b)

# Plotting actual vs predicted weight loss
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5, label="Predicted vs Actual")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r', linestyle="dashed", label="Ideal Fit")
plt.xlabel("Actual Weight Loss")
plt.ylabel("Predicted Weight Loss")
plt.title("Actual vs Predicted Weight Loss")
plt.legend()
plt.show()

