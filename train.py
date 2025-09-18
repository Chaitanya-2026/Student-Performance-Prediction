import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# Load dataset (CSV in /data folder)
data_path = "data/student-mat.csv"
df = pd.read_csv(data_path)

X = df.drop(columns=["G3"])  # G3 = Final Grade
y = df["G3"]

# classification (Pass/Fail)
y = np.where(y >= 10, 1, 0)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {acc*100:.2f}%")

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/student_model.pkl")
print("Model saved at models/student_model.pkl")

