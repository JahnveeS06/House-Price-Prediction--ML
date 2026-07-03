import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import joblib

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor

df= pd.read_csv("data/encoded_house_prices.csv")

# X & y split
X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]

# test-train split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# No need of feature scaling

# Train Models

# 1. Decision Tree Regressor
decision_tree = DecisionTreeRegressor(random_state=42)
decision_tree.fit(X_train, y_train)

# 2. Random Forest Regressor
random_forest = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)
random_forest.fit(X_train, y_train)

# 3. Gradient Boosting
gradient_boost = GradientBoostingRegressor(random_state=42)
gradient_boost.fit(X_train, y_train)

# 4. XGBRegressor
xgboost = XGBRegressor(
    random_state=42,
    eval_metric="rmse"
)
xgboost.fit(X_train, y_train)

# Predict values
dt_pred = decision_tree.predict(X_test)

rf_pred = random_forest.predict(X_test)

gb_pred = gradient_boost.predict(X_test)

xgb_pred = xgboost.predict(X_test)

# Metric evaluation
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print(f"\n{model_name}")
    print("-" * 30)
    print(f"MAE : {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²  : {r2:.4f}")

evaluate_model(y_test, dt_pred, "Decision Tree")
evaluate_model(y_test, rf_pred, "Random Forest")
evaluate_model(y_test, gb_pred, "Gradient Boosting")
evaluate_model(y_test, xgb_pred, "XGBoost")

print("\nComparison Complete!")
print("Compare the R² scores to determine the best-performing model.")

joblib.dump(gradient_boost, "models/gradient_boost.pkl")