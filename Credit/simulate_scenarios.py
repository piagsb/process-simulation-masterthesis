import pandas as pd

# Load file `Szenario-Vergleich.csv`
data = pd.read_csv('Szenario-Vergleich.csv')
print(data.columns)

# Convert `amClass` to numeric values
am_class_mapping = {'low': 0, 'medium': 1, 'high': 2}
data['amClass'] = data['amClass'].map(am_class_mapping)

# Define cost estimates
call_cost = 10
missing_info_cost = 50
fix_cost = 100
opportunity_cost_per_second = 0.05
profit_mapping = {0: 500, 1: 1500, 2: 3000}

# calculate profitability
data['profit'] = data['amClass'].map(profit_mapping) - (
    (data['miss#'] * missing_info_cost) +
    (data['fix'] * fix_cost) +
    (data['duration'] * opportunity_cost_per_second)
) - (data['call#'] * call_cost)

# calculate Rework-Rate
data['rework_rate'] = (data['miss#'] + data['fix']) / (data['miss#'] + data['fix'] + data['offer#'] + data['reply#'])

# save values in `simulated_scenarios.csv`
data.to_csv('simulated_scenarios.csv', index=False)

# print summary of optimized scenarios
summary = data[['amClass', 'profit', 'rework_rate']].groupby('amClass').agg({'profit': 'mean', 'rework_rate': 'mean'})
print(summary)
