"""Train, compare and evaluate customer churn classifiers."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    RocCurveDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "Telco-Customer-Churn.csv"
REPORTS = ROOT / "reports"


def make_preprocessor(numeric_columns: list[str], categorical_columns: list[str]):
    """Impute numeric/categorical gaps and encode categories for scikit-learn."""
    numeric_steps = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_steps = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_steps, numeric_columns),
            ("categorical", categorical_steps, categorical_columns),
        ]
    )


def load_customers() -> tuple[pd.DataFrame, pd.Series]:
    """Load the sample and turn its Yes/No churn label into 1/0."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Customer dataset not found: {DATA_PATH}")

    customers = pd.read_csv(DATA_PATH)
    if "Churn" not in customers.columns:
        raise ValueError("Dataset must contain a Churn column.")
    customers = customers.drop(columns=["customerID"], errors="ignore")
    customers["TotalCharges"] = pd.to_numeric(
        customers["TotalCharges"], errors="coerce"
    )

    target = customers.pop("Churn").map({"No": 0, "Yes": 1})
    if target.isna().any():
        raise ValueError("Churn must contain only Yes/No values.")
    return customers, target.astype(int)


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    customers, target = load_customers()

    numeric_columns = customers.select_dtypes(include="number").columns.tolist()
    categorical_columns = customers.select_dtypes(
        exclude="number"
    ).columns.tolist()
    X_train, X_test, y_train, y_test = train_test_split(
        customers,
        target,
        test_size=0.20,
        random_state=42,
        stratify=target,
    )

    # Keep the preprocessor separate for each pipeline so each model owns its
    # fitted imputers and category encoder.
    estimators = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }

    fitted_models = {}
    results = {}
    for name, estimator in estimators.items():
        pipeline = Pipeline(
            [
                (
                    "preprocess",
                    make_preprocessor(numeric_columns, categorical_columns),
                ),
                ("model", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)
        churn_probabilities = pipeline.predict_proba(X_test)[:, 1]
        predictions = (churn_probabilities >= 0.50).astype(int)
        matrix = confusion_matrix(y_test, predictions, labels=[0, 1])

        fitted_models[name] = pipeline
        results[name] = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, zero_division=0),
            "recall": recall_score(y_test, predictions, zero_division=0),
            "f1": f1_score(y_test, predictions, zero_division=0),
            "roc_auc": roc_auc_score(y_test, churn_probabilities),
            "confusion_matrix": matrix,
        }

    # ROC-AUC evaluates how well the models rank risk over different thresholds.
    best_name = max(results, key=lambda name: results[name]["roc_auc"])
    best_model = fitted_models[best_name]
    dump(best_model, REPORTS / "churn_pipeline.joblib")

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name, pipeline in fitted_models.items():
        RocCurveDisplay.from_estimator(
            pipeline,
            X_test,
            y_test,
            name=name,
            ax=axes[0],
        )
    axes[0].set_title("ROC curves — test set")

    ConfusionMatrixDisplay(
        confusion_matrix=results[best_name]["confusion_matrix"],
        display_labels=["Stay", "Churn"],
    ).plot(ax=axes[1], colorbar=False, cmap="Blues", values_format="d")
    axes[1].set_title(f"{best_name} — test set")
    figure.tight_layout()
    figure.savefig(REPORTS / "evaluation.png", dpi=160)
    plt.close(figure)

    # Save a top-feature chart for either selected model. Logistic coefficients
    # include direction; Random Forest importances show relative contribution.
    selected_estimator = best_model.named_steps["model"]
    feature_names = best_model.named_steps["preprocess"].get_feature_names_out()
    if hasattr(selected_estimator, "feature_importances_"):
        values = selected_estimator.feature_importances_
        importance_label = "Random Forest importance"
        chart_title = "Top churn prediction features — Random Forest"
    else:
        values = selected_estimator.coef_.ravel()
        importance_label = "Logistic Regression coefficient"
        chart_title = "Top churn prediction features — Logistic Regression"

    feature_effects = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_or_coefficient": values,
        }
    )
    feature_effects["feature"] = (
        feature_effects["feature"]
        .str.replace("numeric__", "", regex=False)
        .str.replace("categorical__", "", regex=False)
    )
    top_features = (
        feature_effects.assign(
            absolute_effect=feature_effects["importance_or_coefficient"].abs()
        )
        .nlargest(15, "absolute_effect")
        .sort_values("importance_or_coefficient")
    )
    top_features.drop(columns="absolute_effect").to_csv(
        REPORTS / "top_churn_factors.csv",
        index=False,
    )
    top_features.plot.barh(
        x="feature",
        y="importance_or_coefficient",
        legend=False,
        figsize=(9, 6),
        color="#3973ac",
    )
    plt.title(chart_title)
    plt.xlabel(importance_label)
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(REPORTS / "top_churn_factors.png", dpi=160)
    plt.close()

    report_lines = [
        "Dataset: IBM Telco Customer Churn sample",
        f"Customer records: {len(customers)}",
        f"Churn prevalence: {target.mean():.1%}",
        f"Train/test split: {len(X_train)} / {len(X_test)} (stratified)",
        "Classification threshold: 0.50",
        f"Selected model (highest test ROC-AUC): {best_name}",
    ]
    for name, result in results.items():
        report_lines.extend(
            [
                "",
                name,
                f"Accuracy: {result['accuracy']:.4f}",
                f"Precision (churn): {result['precision']:.4f}",
                f"Recall (churn): {result['recall']:.4f}",
                f"F1-score (churn): {result['f1']:.4f}",
                f"ROC-AUC: {result['roc_auc']:.4f}",
                "Confusion matrix [[TN, FP], [FN, TP]]:",
                str(result["confusion_matrix"]),
            ]
        )

    report = "\n".join(report_lines)
    (REPORTS / "metrics.txt").write_text(report, encoding="utf-8")
    print(report)
    print(f"\nSaved model and charts in: {REPORTS}")


if __name__ == "__main__":
    main()
