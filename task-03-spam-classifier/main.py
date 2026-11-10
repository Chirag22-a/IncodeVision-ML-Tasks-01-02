"""Train and evaluate a simple SMS spam classifier."""

from pathlib import Path
import csv

import matplotlib.pyplot as plt
import pandas as pd
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "uci" / "SMSSpamCollection"
REPORTS = ROOT / "reports"


def load_messages() -> pd.DataFrame:
    """Read the tab-separated UCI messages and check their expected columns."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"SMS dataset not found: {DATA_PATH}")

    messages = pd.read_csv(
        DATA_PATH,
        sep="\t",
        header=None,
        names=["label", "message"],
        encoding="latin-1",
        quoting=csv.QUOTE_NONE,
    )
    if messages.empty or messages[["label", "message"]].isna().any().any():
        raise ValueError("The SMS dataset is empty or has missing labels/messages.")
    if not set(messages["label"].unique()).issubset({"ham", "spam"}):
        raise ValueError("Expected labels to be 'ham' and 'spam'.")
    return messages


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    messages = load_messages()

    X_train, X_test, y_train, y_test = train_test_split(
        messages["message"],
        messages["label"],
        test_size=0.20,
        random_state=42,
        stratify=messages["label"],
    )

    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                ),
            ),
            ("classifier", MultinomialNB()),
        ]
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    labels = ["ham", "spam"]
    matrix = confusion_matrix(y_test, predictions, labels=labels)
    metrics = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision (spam)": precision_score(
            y_test, predictions, pos_label="spam", zero_division=0
        ),
        "Recall (spam)": recall_score(
            y_test, predictions, pos_label="spam", zero_division=0
        ),
        "F1-score (spam)": f1_score(
            y_test, predictions, pos_label="spam", zero_division=0
        ),
    }

    dump(model, REPORTS / "spam_pipeline.joblib")

    matrix_display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Not spam", "Spam"],
    )
    matrix_display.plot(cmap="Blues", values_format="d", colorbar=False)
    plt.title("SMS spam classifier — test set")
    plt.tight_layout()
    plt.savefig(REPORTS / "confusion_matrix.png", dpi=160)
    plt.close()

    # Show the words and short phrases that the model most associates with spam.
    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]
    spam_index = list(classifier.classes_).index("spam")
    ham_index = list(classifier.classes_).index("ham")
    spam_scores = classifier.feature_log_prob_[spam_index] - classifier.feature_log_prob_[ham_index]
    top_indices = spam_scores.argsort()[-15:]
    top_terms = pd.Series(
        spam_scores[top_indices],
        index=vectorizer.get_feature_names_out()[top_indices],
    ).sort_values()
    top_terms.plot.barh(figsize=(8, 6), color="#d77a28")
    plt.title("Terms most associated with spam")
    plt.xlabel("Difference in log probability (spam minus ham)")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(REPORTS / "top_spam_terms.png", dpi=160)
    plt.close()

    report_lines = [
        f"Dataset: UCI SMS Spam Collection",
        f"Messages: {len(messages)}",
        f"Training messages: {len(X_train)}",
        f"Test messages: {len(X_test)}",
        "Model: TF-IDF (unigrams and bigrams) + Multinomial Naive Bayes",
        "",
    ]
    report_lines.extend(f"{name}: {value:.4f}" for name, value in metrics.items())
    report_lines.extend(
        [
            "",
            "Confusion matrix [rows=true, columns=predicted; ham, spam]:",
            str(matrix),
            "",
            "Detailed classification report:",
            classification_report(
                y_test,
                predictions,
                labels=labels,
                target_names=["Not spam (ham)", "Spam"],
                zero_division=0,
            ),
        ]
    )
    report = "\n".join(report_lines)
    (REPORTS / "metrics.txt").write_text(report, encoding="utf-8")
    print(report)
    print(f"\nSaved model and charts in: {REPORTS}")


if __name__ == "__main__":
    main()
