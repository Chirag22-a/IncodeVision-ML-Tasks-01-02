# Task 02 - Linear Regression House Prices

Loads the California Housing dataset, imputes missing numeric values, makes a reproducible train/test split, trains a scaled multiple Linear Regression pipeline and evaluates MSE, RMSE, MAE and R-squared. It saves actual-vs-predicted and residual plots plus a short report.

## Run
`pip install -r requirements.txt` then `python main.py`. First run downloads and caches the dataset via scikit-learn; internet is needed once. Outputs go to `reports/`.

## Data and interpretation
The scikit-learn California Housing sample has 20,640 1990 California census block-group observations and eight features. Target values are median house value in units of $100,000, capped in the source at $500,000. This is historical district-level data, not present-day listing prices. Source: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html .
