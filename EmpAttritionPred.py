import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load HR dataset
data = pd.read_csv("datasets/Employee-Attrition.csv")

# Simple preprocessing: Convert categorical → numeric
data = pd.get_dummies(data, drop_first=True)

# Features and target
X = data.drop("Attrition_Yes", axis=1)

# Target column must be adjusted as per dataset
y = data["Attrition_Yes"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
