"""Classify a message after running main.py once to train the model."""

import argparse
from pathlib import Path

from joblib import load


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "reports" / "spam_pipeline.joblib"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify a text message as spam or not spam."
    )
    parser.add_argument(
        "message",
        nargs="+",
        help="The message text. Put it in quotes if it has spaces.",
    )
    args = parser.parse_args()
    message = " ".join(args.message)

    if not MODEL_PATH.exists():
        parser.error("Model not found. Run python main.py first.")

    model = load(MODEL_PATH)
    prediction = model.predict([message])[0]
    probabilities = model.predict_proba([message])[0]
    classes = model.named_steps["classifier"].classes_
    spam_probability = probabilities[list(classes).index("spam")]

    label = "SPAM" if prediction == "spam" else "NOT SPAM (ham)"
    print(f"Prediction: {label}")
    print(f"Estimated spam probability: {spam_probability:.1%}")


if __name__ == "__main__":
    main()
