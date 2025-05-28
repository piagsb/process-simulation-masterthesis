import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
import pickle

# 1. Daten laden
df = pd.read_csv("traffic_final_for_training.csv")

# 2. Features & Targets definieren
df = df.drop(columns=["duration_so_far"])

# 2. Features & Targets definieren
X = df.drop(columns=["case_id", "timestamp", "duration_days", "received_credit"])
    #One-Hot-Encoding für kategoriale Spalten
X = pd.get_dummies(X, drop_first=True)

y_duration = df["duration_days"]
y_credit = df["received_credit"]

# 3. Train-Test-Split
X_train, X_test, y_train_dur, y_test_dur = train_test_split(X, y_duration, test_size=0.2, random_state=42)
_, _, y_train_cred, y_test_cred = train_test_split(X, y_credit, test_size=0.2, random_state=42)

# 3.1. Modelle initialisieren
model_dur = RandomForestRegressor(random_state=42)
model_cred = RandomForestRegressor(random_state=42)

# Cross-Validation für beide Modelle evaluieren
print("\n📊 Cross-Validation Ergebnisse:")

# Cross-Validation für Dauer-Vorhersage
cv_scores_dur = cross_val_score(model_dur, X_train, y_train_dur, cv=5, scoring="r2")
print("⏱️ Dauer (R², 5-fold):", cv_scores_dur)
print("⏱️ Durchschnittlicher R²:", np.mean(cv_scores_dur))

# Cross-Validation für Credit-Vorhersage
cv_scores_cred = cross_val_score(model_cred, X_train, y_train_cred, cv=5, scoring="r2")
print("\n💰 Credit (R², 5-fold):", cv_scores_cred)
print("💰 Durchschnittlicher R²:", np.mean(cv_scores_cred))

# 4. Modelle trainieren
model_dur.fit(X_train, y_train_dur)
model_cred.fit(X_train, y_train_cred)

# 5. Evaluation
y_pred_dur = model_dur.predict(X_test)
y_pred_cred = model_cred.predict(X_test)

print("⏱️ Dauer-Vorhersage:")
print("MAE:", mean_absolute_error(y_test_dur, y_pred_dur))
print("R²:", r2_score(y_test_dur, y_pred_dur))

print("\n💰 Credit-Vorhersage:")
print("MAE:", mean_absolute_error(y_test_cred, y_pred_cred))
print("R²:", r2_score(y_test_cred, y_pred_cred))

# 6. Speichern
with open("traffic_model_duration.pkl", "wb") as f:
    pickle.dump(model_dur, f)

with open("traffic_model_credit.pkl", "wb") as f:
    pickle.dump(model_cred, f)

print("\n✅ Modelle gespeichert!")

#Feature Importance Analyse

import matplotlib.pyplot as plt
import seaborn as sns

# Feature Importance für Dauer-Vorhersage
importances_dur = model_dur.feature_importances_
features_dur = X.columns
indices_dur = np.argsort(importances_dur)[::-1]

plt.figure(figsize=(10, 6))
sns.barplot(x=importances_dur[indices_dur][:10], y=features_dur[indices_dur][:10])
plt.title("🔍 Wichtigste Features – Dauer (process time)")
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.tight_layout()
plt.show()

# Feature Importance für Credit-Vorhersage
importances_cred = model_cred.feature_importances_
features_cred = X.columns
indices_cred = np.argsort(importances_cred)[::-1]

plt.figure(figsize=(10, 6))
sns.barplot(x=importances_cred[indices_cred][:10], y=features_cred[indices_cred][:10])
plt.title("💰 Wichtigste Features – Received Credit")
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.tight_layout()
plt.show()

# Nach dem Training – Feature-Spalten speichern
with open("traffic_feature_columns.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)
print("✅ Verwendete Feature-Spalten gespeichert als 'traffic_feature_columns.pkl'")