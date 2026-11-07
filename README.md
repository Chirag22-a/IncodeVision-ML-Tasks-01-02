# Task 01 - Data Cleaning and Visualization

Cleans the open Titanic passenger dataset: reports missing values and duplicates, imputes numeric/categorical fields, fixes types, flags extreme numeric values with the IQR rule, and produces EDA plots and a cleaned CSV.

## Run
`pip install -r requirements.txt` then `python main.py`. Outputs go to `reports/`.

## Data
`data/titanic.csv` is the seaborn example copy of the Titanic data: https://github.com/mwaskom/seaborn-data/blob/master/titanic.csv . It is a teaching dataset with passenger-level records, missing fields, and a survival label; it is not a complete manifest of everyone aboard. Outliers are flagged for inspection, not automatically deleted.
