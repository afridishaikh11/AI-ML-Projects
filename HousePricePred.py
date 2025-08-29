from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing(as_frame=True)
data = housing.frame

print(data.head())

# Features (X) and Target (y)
X = data.drop("MedHouseVal", axis=1)
y = data["MedHouseVal"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Build Model
model = Sequential([
    Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dense(1)   # Regression → 1 neuron, no activation
])

model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# Train Model
history = model.fit(X_train, y_train, validation_split=0.2, epochs=30, batch_size=32, verbose=1)

# Evaluate
loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Test MAE: {mae:.2f}")

# Predict
y_pred = model.predict(X_test[:5])
print("Predicted house prices:", y_pred.flatten())
print("Actual house prices:", y_test[:5].values)
