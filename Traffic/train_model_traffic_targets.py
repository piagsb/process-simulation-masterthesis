import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, r2_score

# load data
df = pd.read_csv("traffic_targets.csv")
print("colmns:", df.columns.tolist())

# One-Hot-Encoding
categorical_cols = ['activity']  # ggf. erweitern
df = pd.get_dummies(df, columns=categorical_cols)

# Feature-Definition
print("distribution of target variable 'success':")
print(df['success'].value_counts(normalize=True))
features = [col for col in df.columns if col not in ['case_id', 'success', 'received_credit', 'timestamp']]
X = df[features]
y_success = df['success']
y_credits = df['received_credit']

# Train-Test-Split
X_train, X_test, y_train_succ, y_test_succ = train_test_split(X, y_success, test_size=0.2, random_state=42)
_, _, y_train_cred, y_test_cred = train_test_split(X, y_credits, test_size=0.2, random_state=42)

# initialize models
model_success = RandomForestClassifier(random_state=42, class_weight='balanced')
model_credits = RandomForestRegressor(random_state=42)

# Cross-Validation
print("Cross-Validation Results:")
cv_scores_succ = cross_val_score(model_success, X, y_success, cv=5, scoring="accuracy")
print("Accuracy, 5-fold):", cv_scores_succ)
print("Avg Accuracy:", np.mean(cv_scores_succ))

cv_scores_cred = cross_val_score(model_credits, X, y_credits, cv=5, scoring="r2")
print("Credits (R², 5-fold):", cv_scores_cred)
print("Avg R²:", np.mean(cv_scores_cred))

# Train models
model_success.fit(X_train, y_train_succ)
model_credits.fit(X_train, y_train_cred)

# Save features for later use
with open("traffic_feature_columns.pkl", "wb") as f:
    pickle.dump(features, f)

# Evaluate performance on test data
y_pred_succ = model_success.predict(X_test)
print("\nSuccess (Test data):")
print("Accuracy:", accuracy_score(y_test_succ, y_pred_succ))
print(classification_report(y_test_succ, y_pred_succ))

y_pred_cred = model_credits.predict(X_test)
print("\nCredits (Test data):")
print("MAE:", mean_absolute_error(y_test_cred, y_pred_cred))
print("R²:", r2_score(y_test_cred, y_pred_cred))

# Save models
with open("traffic_model_success.pkl", "wb") as f:
    pickle.dump(model_success, f)
with open("traffic_model_credits.pkl", "wb") as f:
    pickle.dump(model_credits, f)
print("\nModels saved successfully.")

# Analyze Feature Importance
import matplotlib.pyplot as plt
import seaborn as sns

# Feature Importance für Classification-Model
if hasattr(model_success, "feature_importances_"):
    importance_df_succ = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model_success.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    print("\nFeature Importance (Success Model):")
    print(importance_df_succ)

    plt.figure(figsize=(8, 5))
    sns.barplot(x="Importance", y="Feature", data=importance_df_succ)
    plt.title("Feature Importance – Erfolg (Success)")
    plt.tight_layout()
    plt.show()

# Feature Importance for Regression model
if hasattr(model_credits, "feature_importances_"):
    importance_df_cred = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model_credits.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    print("\nFeature Importance (Credit-Model):")
    print(importance_df_cred)

    plt.figure(figsize=(8, 5))
    sns.barplot(x="Importance", y="Feature", data=importance_df_cred)
    plt.title("Feature Importance – Credits")
    plt.tight_layout()
    plt.show()

# Feature Correlation Analysis
corr_matrix = df[features + ['success', 'received_credit']].corr(numeric_only=True)
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation matrix")
plt.tight_layout()
plt.show()