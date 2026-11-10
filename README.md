# IncodeVision ML Internship Tasks (01–04)

Four self-contained Python projects based on the IncodeVision internship task briefs. Each task folder includes its own README, source code, requirements, dataset, and generated reports.

## Task 01: Data Cleaning and Visualization
Cleans and explores the Titanic teaching dataset. It handles missing values and duplicates, fixes selected data types, flags outliers for review, and saves a cleaned CSV, summary, and EDA charts.

## Task 02: Linear Regression House Prices
Trains a scikit-learn Linear Regression pipeline on the California Housing dataset, evaluates a held-out test set, and saves metrics and diagnostic plots.

## Task 03: Spam Message Classifier
Trains a TF-IDF and Multinomial Naive Bayes classifier on the UCI SMS Spam Collection. It reports accuracy, precision, recall, F1-score and a confusion matrix, and includes a command-line message predictor. The source task slide says email; the included public dataset contains SMS, so the project is a text-message demonstration rather than an email-validated model.

## Task 04: Customer Churn Prediction
Compares Logistic Regression and Random Forest on the IBM Telco sample, selects by test ROC-AUC, and saves evaluation metrics, ROC/confusion-matrix plots, and leading churn factors. An optional Streamlit form accepts a customer profile.

## Run a task
Open PowerShell in the task folder, then create an environment, install dependencies, and run the training script:

    py -m venv .venv
    .\.venv\Scripts\python.exe -m pip install -r requirements.txt
    .\.venv\Scripts\python.exe main.py

Task 2 downloads the California Housing sample on its first run. Task 3 and Task 4 save model files in reports when trained; generated model binaries are not included in this repository. See each task README for dataset sources, limitations, prediction instructions, and the optional Task 4 app.
