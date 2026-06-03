# ==========================================
# TASK 3 - MODEL API DEMO
# ==========================================

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import json

# Load Dataset
iris = load_iris()

X = iris.data
y = iris.target

# Train Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

print("Model Trained Successfully!")

# Example API Request
request_data = {
    "features": [5.1, 3.5, 1.4, 0.2]
}

print("\nSample Request:")
print(json.dumps(request_data, indent=2))

# Prediction
prediction = model.predict(
    [request_data["features"]]
)

response = {
    "prediction_value": int(prediction[0]),
    "prediction_class": iris.target_names[prediction[0]]
}

print("\nSample Response:")
print(json.dumps(response, indent=2))