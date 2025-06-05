# Process Simulation Framework – Master's Thesis

This repository contains the prototype developed for the master's thesis:  
**"Prescriptive Business Process Simulation using Supervised Learning"**

## Project Structure

The project is divided into two domains:

- `Credit/`: Simulation and optimization of a loan application scenario (BPI Challenge 2017)
- `Traffic/`: Analysis and optimization of traffic violation cases (Road Traffic Fine Management Process)

Each folder contains:
- Dataset (excluded from GitHub due to size constraints, need to be added manually):
- Credit: https://data.4tu.nl/articles/_/12696884/1
- Traffic: https://data.4tu.nl/articles/_/12683249/1
- Feature engineering scripts
- Model training (Random Forest)
- Activity recommendation and scenario simulation

## Objective

The framework uses **supervised machine learning** to predict process outcomes, such as binary success or numeric KPIs (e.g. payment amount, rework rate).  
Based on these predictions, the system recommends **the most promising next activity** to optimize the desired KPI.

## Setup

Python 3.9 environment is required. Install dependencies via:

```bash
pip install -r requirements.txt
```

Key libraries used include:
- pandas, numpy, scikit-learn
- matplotlib, seaborn

## Dataset Information

The original `.xes` event logs are not included in this repository due to GitHub's 100MB file size limit.  
However, all scripts are built to run on locally available data in CSV format that has been preprocessed accordingly.

## Reproducibility

The pipeline is modular. To reproduce the results for the credit dataset:

```bash
cd Credit/
python preprocess_data.py
python feature_engineering.py
python train_model.py
python recommended_next_activity.py
```

The same applies to the traffic dataset, found in the `Traffic/` folder.

## About the Thesis

This project was developed as part of a master's thesis in Information Systems at TU Berlin.  
The objective was to explore how traditional process simulation can be enhanced through predictive and prescriptive models using supervised learning.

