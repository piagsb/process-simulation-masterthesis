import pandas as pd
import pickle
import numpy as np

# Load file
df = pd.read_csv("traffic_features_engineered_with_targets.csv")

# Load models
with open("traffic_model_duration.pkl", "rb") as f:
    model_dur = pickle.load(f)
with open("traffic_model_credit.pkl", "rb") as f:
    model_cred = pickle.load(f)

# Features used in training
with open("traffic_feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

# Choose case sample
case_id = "A100"
case_data = df[df["case_id"] == case_id].copy()
last_known = case_data.iloc[-1:]
last_activity = last_known["activity"].values[0]
print(f"\nLast known activity: {last_activity}")

# Potencial next activities
possible_next_activities = [
    "Send Appeal to Prefecture",
    "Receive Result Appeal from Prefecture",
    "Add Penalty",
    "Payment",
    "Notify Result Appeal to Offender",
    "Send for Credit Collection"
]

print("\nEvaluation of potential next activities:")

# Last known index and time
base_event_index = last_known["event_index"].values[0]
duration_so_far = last_known["duration_so_far"].values[0]
cum_amount = last_known["cumulative_amount"].values[0]
cum_payment = last_known["cumulative_payment"].values[0]
payment_count = last_known["payment_count"].values[0]
penalty_count = last_known["add_penalty_count"].values[0]
org_resource = last_known["org_resource"].values[0]

for activity in possible_next_activities:
    sim_data = pd.DataFrame([{
        "activity": activity,
        "event_index": base_event_index + 1,
        "duration_so_far": duration_so_far,
        "cumulative_amount": cum_amount + (50 if activity == "Add Penalty" else 0),
        "cumulative_payment": cum_payment + (cum_amount if activity == "Payment" else 0),
        "payment_count": payment_count + (1 if activity == "Payment" else 0),
        "add_penalty_count": penalty_count + (1 if activity == "Add Penalty" else 0),
        "org_resource": org_resource
    }])

    # One-Hot-Encoding and sort
    X_sim = pd.get_dummies(sim_data, columns=["activity", "org_resource"], dtype=float)
    for col in feature_columns:
        if col not in X_sim.columns:
            X_sim[col] = 0
    X_sim = X_sim[feature_columns]

    # prediction
    pred_duration = model_dur.predict(X_sim)[0]
    pred_credits = model_cred.predict(X_sim)[0]

    print(f"→ {activity}: Duration = {pred_duration:.2f}, Credits = {pred_credits:.2f}")
