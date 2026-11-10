# Task 04 — Customer Churn Prediction

This project estimates which customers in a sample telecom dataset are more likely to leave. It imputes missing values, one-hot encodes categories, compares Logistic Regression and Random Forest, and selects the model with the higher test ROC-AUC. The saved pipeline includes all preprocessing steps.

## Train on Windows

Open PowerShell in this task folder, then run:

    py -m venv .venv
    .\.venv\Scripts\python.exe -m pip install -r requirements.txt
    .\.venv\Scripts\python.exe main.py

The training command saves these files in the reports folder:

- metrics.txt — accuracy, precision, recall, F1-score, ROC-AUC and confusion matrices
- evaluation.png — ROC curves and the selected model's confusion matrix
- top_churn_factors.png and top_churn_factors.csv — leading factors in the selected model
- churn_pipeline.joblib — trained preprocessing and prediction pipeline

## Try a customer profile

After training, launch the optional form:

    .\.venv\Scripts\python.exe -m streamlit run app.py

Enter a sample customer profile in the browser page that opens. Stop the local app with Ctrl+C in PowerShell.

## Dataset and limits

The included IBM Telco sample has 7,043 fictional customer records. Source: https://github.com/IBM/telco-customer-churn-on-icp4d . It is a cross-sectional teaching dataset, not a production retention system or a forecast of a specific customer's future. Risk scores are estimates and do not show that an intervention will prevent churn.
