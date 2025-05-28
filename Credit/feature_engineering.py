import pandas as pd

def load_data(file_path):
    """Loads the preprocessed dataset."""
    df = pd.read_csv(file_path)
    return df

def feature_engineering(df):
    """Extracts relevant features for supervised learning."""
    print(df['timestamp'].dtype)

    # Sort data by case_id and timestamp
    df = df.sort_values(by=['case_id', 'timestamp'])

    # Define relevant activities
    offer_events = ['O_Create Offer']
    call_events = ['W_Call after offers']
    miss_events = ['W_Call incomplete files']
    reply_events = ['A_Accepted']
    fix_events = ['W_Validate application']

    # Compute historical feature counts per case
    df['call#'] = df.groupby('case_id')['activity'].transform(lambda x: (x.isin(call_events)).cumsum())
    df['miss#'] = df.groupby('case_id')['activity'].transform(lambda x: (x.isin(miss_events)).cumsum())
    df['offer#'] = df.groupby('case_id')['activity'].transform(lambda x: (x.isin(offer_events)).cumsum())
    df['reply#'] = df.groupby('case_id')['activity'].transform(lambda x: (x.isin(reply_events)).cumsum())
    df['fix'] = df.groupby('case_id')['activity'].transform(lambda x: (x.isin(fix_events)).cumsum() > 0).astype(int)

    # Extract last activity per case
    df['LA'] = df.groupby('case_id')['activity'].transform('last')

    # Compute loan amount class, handling missing values
    df['amClass'] = pd.cut(df['case:RequestedAmount'], bins=[0, 6000, 15000, float('inf')], labels=['low', 'medium', 'high'])
    df['amClass'] = df['amClass'].cat.add_categories('unknown').fillna('unknown') 

    # Calculation of duration within case
    df['duration'] = df.groupby('case_id')['timestamp'].diff().dt.total_seconds()
    df['duration'] = df['duration'].fillna(0) 

    # Compute target variable: whether the loan was granted
    df['granted'] = 0 
    df.loc[df['OfferID'].notna(), 'granted'] = df.groupby('OfferID')['activity'].transform(lambda x: int('O_Accepted' in x.values))
    
    # Replace missing values with 0
    df['granted'] = df['granted'].fillna(0).astype(int)

    # Mapping on case_id (Application-Level) - max() since on Application can have multiple offers
    df['granted'] = df.groupby('case_id')['granted'].transform('max')

    # cost parameters
    cost_per_call = 10
    cost_per_rework = 50
    cost_per_delay = 5
    profit_per_accepted = 500

    # KPI calculation
    df["cost"] = (df["call#"] * cost_per_call) + (df["miss#"] * cost_per_rework) + (df["duration"] * cost_per_delay)
    df["profit"] = df["granted"] * profit_per_accepted - df["cost"]
    df["rework"] = df["miss#"] + df["fix"]

    # combined KPI score
    lambda_profit = 0.7
    lambda_rework = 0.3
    df["score"] = lambda_profit * df["profit"] - lambda_rework * df["rework"]

    # Select relevant columns
    feature_columns = ['case_id', 'LA', 'call#', 'miss#', 'offer#', 'reply#', 'fix', 'amClass', 'duration', 'granted', 'profit', 'rework', 'score']
    df = df[feature_columns]

    return df

if __name__ == "__main__":
    # Load preprocessed data
    file_path = "processed_bpi2017_clean.csv"
    df = load_data(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)

    # Feature engineering
    df = feature_engineering(df)

    # Save processed dataset
    df.to_csv("features_bpi2017.csv", index=False)
    print("Feature engineering complete. Data saved to features_bpi2017.csv")