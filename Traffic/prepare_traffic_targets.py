import pandas as pd

# Load feature file
df = pd.read_csv("traffic_features_engineered.csv")

# Load file with received_credit
df_credits = pd.read_csv("traffic_final_for_training.csv")[['case_id', 'received_credit']]

# translate (success) to total_paid >= total_fine
df['success'] = (df['cumulative_payment'] >= df['cumulative_amount']).astype(int)

# Add feature remaining_amount
df['remaining_amount'] = df['cumulative_amount'] - df['cumulative_payment']

# Add feature has_penalty
df['has_penalty'] = df['add_penalty_count'].apply(lambda x: 1 if x > 0 else 0)

# Add received_credit 
df = df.merge(df_credits, on='case_id', how='left')

# extract target variable
targets = df[['case_id', 'event_index', 'activity', 'success', 'received_credit', 'remaining_amount', 'has_penalty']].copy()

# Save file
targets.to_csv("traffic_targets.csv", index=False)
print("Saved file to traffic_targets.csv")