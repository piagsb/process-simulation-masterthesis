import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
df = pd.read_csv("features_bpi2017.csv")

# define Features & Target-Variables
X = df.drop(columns=["granted", "case_id"])
y = df["granted"]

# One-Hot-Encoding
X = pd.get_dummies(X, columns=["LA", "amClass"], drop_first=True)

# 80% Train, 20% Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# define Hyperparameter range for Randomized Search
param_dist = {
    "n_estimators": [50, 100, 200, 300],
    "max_depth": [10, 20, 30, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", None]
}

rf = RandomForestClassifier(random_state=42)
random_search = RandomizedSearchCV(
    rf, param_distributions=param_dist, n_iter=20, cv=3, scoring="accuracy", n_jobs=-1, random_state=42
)

# Execute Random Search
random_search.fit(X_train, y_train)

best_model = random_search.best_estimator_

# Evaluation of best model
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Optimized Model:\n Best Accuracy: {accuracy:.4f}")
print("\nClassification report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\n Confusion Matrix:\n", conf_matrix)


with open("credit_model_optimized.pkl", "wb") as file:
    pickle.dump(best_model, file)

print("\nOptimized Model saved as 'credit_model_optimized.pkl'")