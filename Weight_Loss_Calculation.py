# # import numpy as np
# # import pandas as pd
# # import matplotlib.pyplot as plt
# # from sklearn.model_selection import train_test_split
# # from sklearn.preprocessing import StandardScaler, PolynomialFeatures
# # from sklearn.linear_model import LinearRegression
# # from sklearn.metrics import mean_squared_error, r2_score

# # # Create realistic sample dataset
# # np.random.seed(42)
# # BMI = np.random.randint(18, 35, 100)
# # Initial_Weight = np.random.randint(55, 120, 100)
# # Exercise_Duration = np.random.randint(15, 90, 100)

# # # Create a more realistic weight loss pattern
# # Weight_Loss = (
# #     0.05 * BMI +
# #     0.1 * Initial_Weight +
# #     0.3 * Exercise_Duration +
# #     np.random.normal(0, 2, 100)  # Adding some noise
# # )

# # # Convert to DataFrame
# # data = {
# #     'BMI': BMI,
# #     'Initial_Weight': Initial_Weight,
# #     'Exercise_Duration': Exercise_Duration,
# #     'Weight_Loss': Weight_Loss
# # }

# # df = pd.DataFrame(data)

# # # Split into training and test sets
# # X = df[['BMI', 'Initial_Weight', 'Exercise_Duration']]
# # y = df['Weight_Loss']

# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # # Scale the features
# # scaler = StandardScaler()
# # X_train_scaled = scaler.fit_transform(X_train)
# # X_test_scaled = scaler.transform(X_test)

# # # Add polynomial features (degree = 2)
# # poly = PolynomialFeatures(degree=2, include_bias=False)
# # X_train_poly = poly.fit_transform(X_train_scaled)
# # X_test_poly = poly.transform(X_test_scaled)

# # # Train the model
# # model = LinearRegression()
# # model.fit(X_train_poly, y_train)

# # # Print learned parameters
# # print("Learned Weights (Coefficients):", model.coef_)
# # print("Learned Bias (Intercept):", model.intercept_)

# # # Predict using the test set
# # y_pred = model.predict(X_test_poly)

# # # Evaluate the model
# # mse = mean_squared_error(y_test, y_pred)
# # r2 = r2_score(y_test, y_pred)

# # print(f"\nMean Squared Error: {mse:.2f}")
# # print(f"R-squared: {r2:.2f}")

# # # Plot actual vs predicted values
# # plt.scatter(y_test, y_pred, color='blue', label='Predictions')
# # plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', label='Perfect Fit')
# # plt.xlabel('Actual Weight Loss')
# # plt.ylabel('Predicted Weight Loss')
# # plt.title('Actual vs Predicted Weight Loss (Realistic Data)')
# # plt.legend()
# # plt.show()

# # # Show feature importance
# # coefficients = pd.DataFrame(model.coef_, poly.get_feature_names_out(), columns=['Coefficient'])
# # coefficients['Absolute_Coefficient'] = coefficients['Coefficient'].abs()
# # coefficients = coefficients.sort_values(by='Absolute_Coefficient', ascending=False)
# # print("\nFeature Importance:\n", coefficients.head(10))

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, PolynomialFeatures
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, r2_score

# # Generate realistic dataset
# np.random.seed(42)
# heights = np.random.randint(150, 190, 200) / 100  # Convert to meters
# weights = np.random.randint(50, 120, 200)
# bmi = weights / (heights ** 2)
# age = np.random.randint(18, 60, 200)
# gender = np.random.choice([0, 1], 200)  # 0 = Female, 1 = Male
# exercise_duration = np.random.randint(15, 90, 200)  # Minutes per day
# num_days = np.random.randint(20, 90, 200)  # Total number of days
# exercise_type = np.random.choice([0, 1, 2], 200)  # 0 = Light, 1 = Cardio, 2 = Strength

# # Assign weights based on scientific studies
# weight_loss = (
#     0.6 * exercise_duration +  # Higher weight for duration
#     0.5 * num_days +  # Higher weight for consistency
#     0.2 * exercise_type +  # Moderate weight for type of exercise
#     0.15 * bmi +  # Lower weight for BMI
#     (-0.25 * age) +  # Slight negative effect of age
#     (0.1 * gender) +  # Small effect for gender
#     np.random.normal(0, 1, 200)  # Adding realistic noise
# )

# # Create DataFrame
# data = pd.DataFrame({
#     'BMI': bmi,
#     'Age': age,
#     'Gender': gender,
#     'Exercise_Duration': exercise_duration,
#     'Num_Days': num_days,
#     'Exercise_Type': exercise_type,
#     'Weight_Loss': weight_loss
# })

# # Split into training and test sets
# X = data[['BMI', 'Age', 'Gender', 'Exercise_Duration', 'Num_Days', 'Exercise_Type']]
# y = data['Weight_Loss']

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Scale features
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # Add polynomial features (degree = 2 for capturing interactions)
# poly = PolynomialFeatures(degree=2, include_bias=False)
# X_train_poly = poly.fit_transform(X_train_scaled)
# X_test_poly = poly.transform(X_test_scaled)

# # Train the model
# model = LinearRegression()
# model.fit(X_train_poly, y_train)

# # Predictions
# y_pred = model.predict(X_test_poly)

# # Evaluate the model
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print(f"Mean Squared Error: {mse:.2f}")
# print(f"R-squared: {r2:.2f}")

# # Plot actual vs predicted
# plt.scatter(y_test, y_pred, color='blue', label='Predictions')
# plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', label='Perfect Fit')
# plt.xlabel('Actual Weight Loss')
# plt.ylabel('Predicted Weight Loss')
# plt.title('Actual vs Predicted Weight Loss')
# plt.legend()
# plt.show()

# BEST RESULTS 
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, PolynomialFeatures
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, r2_score

# # Generate realistic dataset
# np.random.seed(42)
# heights = np.random.randint(150, 190, 200) / 100  # Convert to meters
# weights = np.random.randint(50, 120, 200)
# bmi = weights / (heights ** 2)
# age = np.random.randint(18, 60, 200)
# gender = np.random.choice([0, 1], 200)  # 0 = Female, 1 = Male
# exercise_duration = np.random.randint(15, 90, 200)  # Minutes per day
# num_days = np.random.randint(20, 90, 200)  # Total number of days
# exercise_type = np.random.choice([0, 1, 2], 200)  # 0 = Light, 1 = Cardio, 2 = Strength

# # Assign weights based on scientific studies (adjusted for realism)
# weight_loss = (
#     0.06 * exercise_duration +  # Adjusted for smaller effect
#     0.05 * num_days +  # Adjusted for smaller effect
#     0.02 * exercise_type +
#     0.015 * bmi +
#     (-0.025 * age) + 
#     (0.01 * gender) + 
#     np.random.normal(0, 0.5, 200)  # Reduced noise
# )

# # Create DataFrame
# data = pd.DataFrame({
#     'BMI': bmi,
#     'Age': age,
#     'Gender': gender,
#     'Exercise_Duration': exercise_duration,
#     'Num_Days': num_days,
#     'Exercise_Type': exercise_type,
#     'Weight_Loss': weight_loss
# })

# # Split into training and test sets
# X = data[['BMI', 'Age', 'Gender', 'Exercise_Duration', 'Num_Days', 'Exercise_Type']]
# y = data['Weight_Loss']

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Scale features
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # Add polynomial features (degree = 2 for capturing interactions)
# poly = PolynomialFeatures(degree=2, include_bias=False)
# X_train_poly = poly.fit_transform(X_train_scaled)
# X_test_poly = poly.transform(X_test_scaled)

# # Train the model
# model = LinearRegression()
# model.fit(X_train_poly, y_train)

# # Predictions
# y_pred = model.predict(X_test_poly)

# # Cap unrealistic weight loss predictions
# y_pred = np.clip(y_pred, 0, 15)

# # Evaluate the model
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print(f"Mean Squared Error: {mse:.2f}")
# print(f"R-squared: {r2:.2f}")


# # Define new input
# new_input = pd.DataFrame({
#     'BMI': [25],
#     'Age': [30],
#     'Gender': [0],
#     'Exercise_Duration': [60],
#     'Num_Days': [60],
#     'Exercise_Type': [1]
# })

# # Scale the input
# new_input_scaled = scaler.transform(new_input)

# # Add polynomial features
# new_input_poly = poly.transform(new_input_scaled)

# # Predict the weight loss
# predicted_weight_loss = model.predict(new_input_poly)

# # Cap the predicted weight loss
# predicted_weight_loss = np.clip(predicted_weight_loss, 0, 15)

# print(f"Predicted Weight Loss: {predicted_weight_loss[0]:.2f} kg")

# # Plot actual vs predicted
# plt.scatter(y_test, y_pred, color='blue', label='Predictions')
# plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', label='Perfect Fit')
# plt.xlabel('Actual Weight Loss')
# plt.ylabel('Predicted Weight Loss')
# plt.title('Actual vs Predicted Weight Loss')
# plt.legend()
# plt.show()

# # The learned parameters are :-
# print("Coefficients:", model.coef_)
# print("Intercept:", model.intercept_)

# more accurate RESULT
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

# Generate realistic dataset
np.random.seed(42)
heights = np.random.randint(150, 190, 200) / 100  # Convert to meters
weights = np.random.randint(50, 120, 200)
bmi = weights / (heights ** 2)
age = np.random.randint(18, 60, 200)
gender = np.random.choice([0, 1], 200)  # 0 = Female, 1 = Male
exercise_duration = np.random.randint(15, 90, 200)  # Minutes per day
num_days = np.random.randint(20, 90, 200)  # Total number of days
exercise_type = np.random.choice([0, 1, 2], 200)  # 0 = Light, 1 = Cardio, 2 = Strength

# Assign weights based on realistic impacts
weight_loss = (
    0.06 * exercise_duration +  # Adjusted weight
    0.05 * num_days +
    0.02 * exercise_type +
    0.015 * bmi +
    (-0.025 * age) +
    (0.01 * gender) +
    np.random.normal(0, 0.5, 200)  # Reduced noise for better stability
)

# Create DataFrame
data = pd.DataFrame({
    'BMI': bmi,
    'Age': age,
    'Gender': gender,
    'Exercise_Duration': exercise_duration,
    'Num_Days': num_days,
    'Exercise_Type': exercise_type,
    'Weight_Loss': weight_loss
})

# Split into training and test sets
X = data[['BMI', 'Age', 'Gender', 'Exercise_Duration', 'Num_Days', 'Exercise_Type']]
y = data['Weight_Loss']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Add polynomial features (degree = 1 to simplify and avoid interactions)
poly = PolynomialFeatures(degree=1, include_bias=False)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

# Train the model with Ridge regression (for regularization)
model = Ridge(alpha=0.5)  # Regularization to reduce extreme coefficients
model.fit(X_train_poly, y_train)

# Predictions
y_pred = model.predict(X_test_poly)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")

# Extract coefficients and intercept
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# Plot actual vs predicted
plt.scatter(y_test, y_pred, color='blue', label='Predictions')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', label='Perfect Fit')
plt.xlabel('Actual Weight Loss')
plt.ylabel('Predicted Weight Loss')
plt.title('Actual vs Predicted Weight Loss')
plt.legend()
plt.show()

# Define new input
new_input = pd.DataFrame({
    'BMI': [25],
    'Age': [30],
    'Gender': [0],
    'Exercise_Duration': [60],
    'Num_Days': [60],
    'Exercise_Type': [1]
})

# Scale the input
new_input_scaled = scaler.transform(new_input)

# Add polynomial features
new_input_poly = poly.transform(new_input_scaled)

# Predict the weight loss
predicted_weight_loss = model.predict(new_input_poly)

# Cap the predicted weight loss
predicted_weight_loss = np.clip(predicted_weight_loss, 0, 15)

print(f"Predicted Weight Loss: {predicted_weight_loss[0]:.2f} kg")


