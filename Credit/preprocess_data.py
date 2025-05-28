import pm4py
import pandas as pd

def load_xes(file_path):
    """load XES-file and convert to Pandas DataFrame."""
    event_log = pm4py.read_xes(file_path)
    df = pm4py.convert_to_dataframe(event_log)
    return df

def preprocess_data(df):
    """clean and transform event log for data analysis."""
    
    # rename columns
    df = df.rename(columns={
        'concept:name': 'activity',
        'time:timestamp': 'timestamp',
        'case:concept:name': 'case_id'
    })

    # convert timestamp to datetime (using UTC)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
    print(f"invalid timestamps after converting: {df['timestamp'].isna().sum()}")

    # fill missing values
    df['timestamp'] = df['timestamp'].ffill()  
    df['timestamp'] = df['timestamp'].bfill() 
    print(f"Missing timestamps after filling: {df['timestamp'].isna().sum()}")

    # convert credit amount to float
    for col in ['case:RequestedAmount', 'OfferedAmount']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Sort by case_id and timestamp
    df = df.sort_values(by=['case_id', 'timestamp'])

    return df

if __name__ == "__main__":
    # Load XES-Datei and prepare data
    file_path = "BPI Challenge 2017.xes"
    df = load_xes(file_path)
    df = preprocess_data(df)

    # Save data
    df.to_csv("processed_bpi2017_clean.csv", index=False)
    print("Preprocessing successfully finished. Data saved as 'processed_bpi2017_clean.csv'")