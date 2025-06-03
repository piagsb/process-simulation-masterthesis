import pandas as pd

# Lade Datei
log_df = pd.read_csv("traffic_log.csv")
log_df["time:timestamp"] = pd.to_datetime(log_df["time:timestamp"])
log_df = log_df.sort_values(by=["case:concept:name", "time:timestamp"])

# Ergebnisliste
features_list = []

# Gruppiere nach Prozessinstanz
for case_id, group in log_df.groupby("case:concept:name"):
    group = group.copy()
    group["duration_so_far"] = (group["time:timestamp"] - group["time:timestamp"].iloc[0]).dt.total_seconds()
    group["event_index"] = range(1, len(group) + 1)
    
    group["cumulative_amount"] = group["amount"].fillna(0).cumsum()
    group["cumulative_payment"] = group["paymentAmount"].fillna(0).cumsum()
    group["payment_count"] = group["concept:name"].eq("Payment").cumsum()
    group["add_penalty_count"] = group["concept:name"].eq("Add penalty").cumsum()
    group["remaining_amount"] = group["cumulative_amount"] - group["cumulative_payment"]
    group["has_penalty"] = group["concept:name"].isin(["Add penalty"]).cumsum().gt(0).astype(int)
    group["org_resource"] = group["org:resource"].iloc[0]  # nur bei erstem Event vorhanden
    
    selected = group[[
        "case:concept:name", "concept:name", "event_index", "duration_so_far",
        "cumulative_amount", "cumulative_payment", "payment_count", "add_penalty_count",
        "remaining_amount", "has_penalty",
        "org_resource", "time:timestamp"
    ]].rename(columns={
        "case:concept:name": "case_id",
        "concept:name": "activity",
        "time:timestamp": "timestamp"
    })
    
    features_list.append(selected)

# Kombinieren
features_df = pd.concat(features_list).reset_index(drop=True)

# Speichern
features_df.to_csv("traffic_features_engineered.csv", index=False)
print("✅ Datei gespeichert als traffic_features_engineered.csv")