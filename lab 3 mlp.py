import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
df = pd.read_csv(r"C:\Users\23adsb50\Downloads\heart.csv")
print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())
df = df.fillna(df.mean(numeric_only=True))
TARGET_COLUMN = "output"   
X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
mlp = MLPRegressor(
    hidden_layer_sizes=(32, 16),
    activation="relu",
    solver="adam",
    max_iter=1000,
    random_state=42
)
mlp.fit(X_train_scaled, y_train)
y_pred = mlp.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("\nModel Performance")
print("-----------------")
print(f"Mean Squared Error: {mse:.4f}")
print(f"R² Score: {r2:.4f}")
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color="red")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("MLP Regression: Actual vs Predicted")
plt.grid(True)
plt.show()