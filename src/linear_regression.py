import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
#import joblib
#from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# split fature(X) and target(y)
# train-test split
# feature scaling

df= pd.read_csv("data/encoded_house_prices.csv")

# X & y split
X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]

# test-train split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# feature scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)  # Fit only on training data
X_test = scaler.transform(X_test)        # Transform test data using the same scaler

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict values
y_pred = model.predict(X_test)

# Metric evaluation
mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

r2 = r2_score(y_test, y_pred)

print("MAE :", mae)
print("RMSE:", rmse)
print("R² Score:", r2)

#For Classification
#print("Accuracy:", accuracy_score(y_test, y_pred))
#print("Precision Recall:", precision_score(y_test, y_pred))
#print("Recall Score:", recall_score(y_test, y_pred))
#print("F1 Score:", f1_score(y_test, y_pred))

# Convert predictions back to original prices
actual_price= np.expm1(y_test)
predicted_price = np.expm1(y_pred)

# Compare actual and predicted house prices
comparison = pd.DataFrame({
    "Actual Price": actual_price,
    "Predicted Price": predicted_price
})

print(comparison.head(10))

# Plot Actual vs Predicted
plt.figure(figsize=(8,6))

plt.scatter(actual_price, predicted_price)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")

plt.plot(
    [actual_price.min(), actual_price.max()],
    [actual_price.min(), actual_price.max()],
    color="red"
)

plt.show()

# Save trained model
#joblib.dump(model, "models/linear_regression.pkl")
#joblib.dump(scaler, "models/scaler.pkl")
# Later, you can load it with:
#model = joblib.load("models/linear_regression.pkl")