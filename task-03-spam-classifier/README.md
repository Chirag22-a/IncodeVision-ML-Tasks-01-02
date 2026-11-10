# Task 03 — Spam Message Classifier

This project trains a text classifier to label a message as spam or ham (not spam). It uses tokenization and English stop-word removal through TF-IDF, followed by Multinomial Naive Bayes. A stratified train/test split keeps the spam/ham balance similar in both sets.

## Run on Windows

Open PowerShell in this task folder, then run:

    py -m venv .venv
    .\.venv\Scripts\python.exe -m pip install -r requirements.txt
    .\.venv\Scripts\python.exe main.py

The training command saves the trained model and results in the reports folder:

- metrics.txt — accuracy, spam precision, spam recall, F1-score and confusion matrix
- confusion_matrix.png — test-set confusion matrix
- top_spam_terms.png — terms the model associates most with spam
- spam_pipeline.joblib — trained TF-IDF and classifier pipeline

Try a new message after training:

    .\.venv\Scripts\python.exe predict.py "Congratulations, claim your free prize now"

Replace the quoted example with your own message.

## Dataset and limits

The included UCI SMS Spam Collection contains 5,574 labeled text messages. Source: https://archive.ics.uci.edu/dataset/228/sms+spam+collection . The internship slide says email, while this standard public dataset contains SMS messages. This project demonstrates spam text classification on SMS; its scores do not establish performance on email.
