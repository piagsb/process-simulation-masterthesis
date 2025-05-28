import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, r2_score, confusion_matrix

# load event log
df = pd.read_csv("features_bpi2017.csv")

# define Features & Target-Variable 
X = df.drop(columns=["score", "case_id"])  # score ist the target, case_id is only an identifier
y = df["score"]

# One-Hot-Encoding of categorical variables
X = pd.get_dummies(X, columns=["LA", "amClass"], drop_first=True)

# Trainings- & Testset split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

#calculate accuracy based on training data
y_train_pred = model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)

print(f"accuracy with training set: {train_accuracy:.4f}")
###

# evaluation
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"model performance:\n MAE: {mae:.4f}\n R² Score: {r2:.4f}\n")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Rejected", "Accepted"], yticklabels=["Rejected", "Accepted"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Feature Importance Analysis
feature_importances = pd.DataFrame({"Feature": X.columns, "Importance": model.feature_importances_})
feature_importances = feature_importances.sort_values(by="Importance", ascending=False)

# Visualize most important features 
plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=feature_importances.head(10))
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("most important features for the process score")
plt.show()

# Save model
with open("credit_model_score.pkl", "wb") as file:
    pickle.dump(model, file)

print("\nModel saved as 'credit_model_score.pkl'")