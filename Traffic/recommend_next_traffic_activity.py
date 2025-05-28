import pandas as pd
import pickle
import numpy as np

# load data
df = pd.read_csv("traffic_targets.csv")

# load models
with open("traffic_model_success.pkl", "rb") as f:
    model_succ = pickle.load(f)
with open("traffic_model_credits.pkl", "rb") as f:
    model_cred = pickle.load(f)

# features used in training
with open("traffic_feature_columns.pkl", "rb") as f:
    training_features = pickle.load(f)

# Ensure "remaining_amount" und "has_penalty" exist in training_features
for col in ["remaining_amount", "has_penalty"]:
    if col in df.columns and col not in training_features:
        training_features.append(col)

def simulate_recommendations_at_stages(case_id, df, model_succ, model_cred, training_features):
    case_data = df[df["case_id"] == case_id].copy()
    max_index = case_data["event_index"].max()
    stages = [0.25, 0.5, 0.75]
    possible_next_activities = [
        "Send Appeal to Prefecture",
        "Receive Result Appeal from Prefecture",
        "Add Penalty",
        "Notify Result Appeal to Offender",
        "Send for Credit Collection"
    ]

    for stage in stages:
        cutoff = int(np.floor(max_index * stage))
        partial_case = case_data[case_data["event_index"] <= cutoff]
        if partial_case.empty:
            print(f"\nStage {int(stage*100)}%: Keine Daten vorhanden.")
            continue
        last_row = partial_case.iloc[[-1]].copy()

        results = []
        for activity in possible_next_activities:
            simulated_row = last_row.copy()
            simulated_row.loc[:, "activity"] = activity

            # update remaining_amount and has_penalty
            if "remaining_amount" in simulated_row.columns:
                # Dummylogic: if activity is 'Send Fine', reduce remaining_amount by 1
                if activity == "Send Fine":
                    simulated_row["remaining_amount"] = max(simulated_row["remaining_amount"].values[0] - 1, 0)

            if "remaining_amount" in simulated_row.columns and activity == "Add Penalty":
                simulated_row["remaining_amount"] = simulated_row["remaining_amount"].values[0] + 1

            X_sim = pd.get_dummies(simulated_row, columns=["activity"], dtype=float)

            for col in training_features:
                if col not in X_sim.columns:
                    X_sim[col] = 0.0
            X_sim = X_sim[training_features]

            pred_success = model_succ.predict(X_sim)[0]
            pred_credits = model_cred.predict(X_sim)[0]

            results.append((activity, pred_success, pred_credits))

        results = sorted(results, key=lambda x: (-x[1], -x[2]))
        best_activity, best_succ, best_cred = results[0]
        print(f"\nStage {int(stage*100)}%:")
        print(f"Recommended next activity: {best_activity}")
        print(f"Expected success: {best_succ}, Expected credits: {best_cred:.2f}")
        print(f"\nCompare all options:")
        for activity, succ, cred in results:
            marker = "!!!" if succ < 1 else ""
            print(f"→ {activity}: sucess = {succ}, credits = {cred:.2f} {marker}")

# sample process
case_id = "N39033"
case_data = df[df["case_id"] == case_id].copy()

# last known activity
last_activity = case_data["activity"].iloc[-1]
print(f"last logged activity: {last_activity}")

# possible next acitivites
possible_next_activities = [
    "Send Appeal to Prefecture",
    "Receive Result Appeal from Prefecture",
    "Add Penalty",
    "Notify Result Appeal to Offender",
    "Send for Credit Collection"
]

print("\nEvaluation of all possible activities:")

# Current success & sredits
actual_success = case_data["success"].iloc[-1]
actual_credits = case_data["received_credit"].iloc[-1]
print(f"Actual sucess: {actual_success}, Actual credits: {actual_credits:.2f}\n")

results = []

for activity in possible_next_activities:
    simulated_row = case_data.iloc[[-1]].copy()
    simulated_row.loc[:, "activity"] = activity

    # update remaining_amount and has_penalty updaten
    if "remaining_amount" in simulated_row.columns:
        # if activity 'Send Fine', reduce remaining_amount by 1
        if activity == "Send Fine":
            simulated_row["remaining_amount"] = max(simulated_row["remaining_amount"].values[0] - 1, 0)

    if "remaining_amount" in simulated_row.columns and activity == "Add Penalty":
        simulated_row["remaining_amount"] = simulated_row["remaining_amount"].values[0] + 1

    # One-Hot-Encoding
    X_sim = pd.get_dummies(simulated_row, columns=["activity"], dtype=float)

    # add missing columns and sort
    for col in training_features:
        if col not in X_sim.columns:
            X_sim[col] = 0.0
    X_sim = X_sim[training_features]

    # prediction of kpis
    pred_success = model_succ.predict(X_sim)[0]
    pred_credits = model_cred.predict(X_sim)[0]

    results.append((activity, pred_success, pred_credits))

# sort results by success
results = sorted(results, key=lambda x: (-x[1], -x[2]))

# display next best activity
best_activity, best_succ, best_cred = results[0]
print(f"Recommended next activity: {best_activity}")
print(f"Expected success: {best_succ}, Expected credits: {best_cred:.2f}")
print("\nComparison of activities:")
for activity, succ, cred in results:
    marker = "!!!" if succ < 1 else ""
    print(f"→ {activity}: Success = {succ}, Credits = {cred:.2f} {marker}")
print(f"\nActual results of {case_id}: Credits {actual_credits:.2f}, Success: {actual_success}")

simulate_recommendations_at_stages("N39033", df, model_succ, model_cred, training_features)