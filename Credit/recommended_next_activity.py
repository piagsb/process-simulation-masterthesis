import pandas as pd
import pickle
import numpy as np
import random

# Load model
with open("credit_model_optimized.pkl", "rb") as file:
    model = pickle.load(file)

# Load features
data = pd.read_csv("features_bpi2017.csv")

# Cost estimates
call_cost = 10
missing_info_cost = 50
fix_cost = 100
opportunity_cost_per_second = 0.05
profit_mapping = {0: 500, 1: 1500, 2: 3000}

# Possible next activities
possible_activities = ["W_Call after offers", "W_Call incomplete files", "W_Validate application", "O_Accepted", "O_Refused"]

def simulate_outcome(case_data, next_activity):
    """Simulate impact of next activity on profit and rework rate."""
    
    
    new_case = case_data.copy()
    
    # simulate impact of activity
    if next_activity == "W_Call after offers":
        new_case["call#"] += 1
    elif next_activity == "W_Call incomplete files":
        new_case["miss#"] += 1
    elif next_activity == "W_Validate application":
        new_case["fix"] = 1
    elif next_activity == "O_Accepted":
        new_case["granted"] = 1
    elif next_activity == "O_Refused":
        new_case["granted"] = 0

    # calculate profit
    amClass_mapping = {"low": 0, "medium": 1, "high": 2, "unknown": 0}
    new_case["profit"] = profit_mapping[amClass_mapping.get(new_case["amClass"], 0)] - (
        (new_case["miss#"] * missing_info_cost) +
        (new_case["fix"] * fix_cost) +
        (new_case["duration"] * opportunity_cost_per_second)
    ) - (new_case["call#"] * call_cost)

    # calculate Rework-Rate
    new_case["rework_rate"] = (new_case["miss#"] + new_case["fix"]) / (new_case["miss#"] + new_case["fix"] + new_case["offer#"] + new_case["reply#"])

    return new_case[["profit", "rework_rate"]]

def recommend_next_activity(case_id):
    """Recommend next best activitiy based on kpi optimization."""
    
    # current case log
    case_data = data[data["case_id"] == case_id].iloc[-1] 

    best_activity = None
    best_profit = -np.inf
    best_rework_rate = np.inf
    profit_threshold = 500  # Profit threshold
    results = []
    
    for activity in possible_activities:
        outcome = simulate_outcome(case_data, activity)
        profit, rework_rate = outcome["profit"], outcome["rework_rate"]
        
        results.append((activity, profit, rework_rate))
        
        # decision logic: Maximize Profit + minimize Rework
        if profit > best_profit or (profit == best_profit and rework_rate < best_rework_rate):
            best_activity = activity
            best_profit = profit
            best_rework_rate = rework_rate
    
    # New rule: if expected Profit under threshold or Rework-Rate over 70% → Cancel case
    if best_profit < profit_threshold or best_rework_rate > 0.7:
        best_activity = "O_Refused"
        best_profit = 0  
        best_rework_rate = 0 

    #Prnt Results
    print(f"best next activity {case_id}: {best_activity}")
    print(f"expected profit: {best_profit}, expected rework-rate: {best_rework_rate}")
    
    return best_activity, best_profit, best_rework_rate

def recommend_next_activity_at_various_stages(case_id):
    """recommend next activity at various stages of the process."""

    case_data_full = data[data["case_id"] == case_id]
    
    # three stages: early, medium, late
    process_stages = [int(len(case_data_full) * 0.25), int(len(case_data_full) * 0.5), int(len(case_data_full) * 0.75)]

    for step in process_stages:
        case_data = case_data_full.iloc[step] 
        
        best_activity, best_profit, best_rework_rate = recommend_next_activity(case_id)

        print(f"recommendation for {case_id} at {step / len(case_data_full) * 100:.0f}% progress:")
        print(f"best next activity: {best_activity}")
        print(f"expected profit: {best_profit:.2f}, expected rework-rate: {best_rework_rate:.4f}")
        print("-" * 50)

def compare_real_vs_model(case_id):
    """Compare real process results with model recommendation by credit class."""
    
    case_data = data[data["case_id"] == case_id]

    # Check if process is finished
    if case_data.iloc[-1]["LA"] not in ["O_Accepted", "O_Refused"]:
        print(f"Case {case_id} not finished. Skip comparison.")
        return None
    
    am_class = case_data.iloc[-1]["amClass"]

    real_profit = profit_mapping.get(am_class, 0) - (
        (case_data["miss#"].sum() * missing_info_cost) +
        (case_data["fix"].sum() * fix_cost) +
        (case_data["duration"].sum() * opportunity_cost_per_second)
    ) - (case_data["call#"].sum() * call_cost)

    real_rework_rate = (case_data["miss#"].sum() + case_data["fix"].sum()) / (
        case_data["miss#"].sum() + case_data["fix"].sum() + case_data["offer#"].sum() + case_data["reply#"].sum()
    )

    best_activity, model_profit, model_rework_rate = recommend_next_activity(case_id)

    return am_class, real_profit, model_profit, real_rework_rate, model_rework_rate

def aggregate_performance_comparison():
    """compare average KPI by credit class."""

    real_profits = {"low": [], "medium": [], "high": []}
    model_profits = {"low": [], "medium": [], "high": []}
    real_rework_rates = {"low": [], "medium": [], "high": []}
    model_rework_rates = {"low": [], "medium": [], "high": []}

    closed_cases = data[data["LA"].isin(["O_Accepted", "O_Refused"])]["case_id"].unique()

    for case_id in closed_cases:
        result = compare_real_vs_model(case_id)
        if result:
            am_class, real_profit, model_profit, real_rework_rate, model_rework_rate = result
            if am_class in real_profits:
                real_profits[am_class].append(real_profit)
                model_profits[am_class].append(model_profit)
                real_rework_rates[am_class].append(real_rework_rate)
                model_rework_rates[am_class].append(model_rework_rate)

    for am_class in ["low", "medium", "high"]:
        avg_real_profit = np.mean(real_profits[am_class]) if real_profits[am_class] else 0
        avg_model_profit = np.mean(model_profits[am_class]) if model_profits[am_class] else 0
        avg_real_rework_rate = np.mean(real_rework_rates[am_class]) if real_rework_rates[am_class] else 0
        avg_model_rework_rate = np.mean(model_rework_rates[am_class]) if model_rework_rates[am_class] else 0

        print(f"Average performance by credit class: {am_class}")
        print(f"Actual avg profit: {avg_real_profit:.2f}, model avg profit: {avg_model_profit:.2f}")
        print(f"Actual avg rework-rate: {avg_real_rework_rate:.4f}, model avg rework-rate: {avg_model_rework_rate:.4f}")
        print("-" * 50)

# test closed cases
aggregate_performance_comparison()