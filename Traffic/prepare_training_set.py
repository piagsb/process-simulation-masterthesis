import pandas as pd

# Load features & target variables
features_df = pd.read_csv("traffic_features_engineered.csv")
targets_df = pd.read_csv("traffic_processed.csv")

# Sort values by last status entry per case 
latest_features = features_df.sort_values(by=["case_id", "timestamp"]).groupby("case_id").tail(1)

# merge with target variables
merged_df = pd.merge(latest_features, targets_df, on="case_id", how="inner")

# Save file
merged_df.to_csv("traffic_final_for_training.csv", index=False)
print("Saved trainings data as traffic_final_for_training.csv")