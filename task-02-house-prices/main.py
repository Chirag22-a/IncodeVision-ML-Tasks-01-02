from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "reports"
OUT.mkdir(exist_ok=True)
data = fetch_california_housing(as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=.2, random_state=42)
model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LinearRegression())
model.fit(X_train, y_train)
pred = model.predict(X_test)
mse = mean_squared_error(y_test, pred); rmse = float(np.sqrt(mse)); mae = mean_absolute_error(y_test, pred); r2 = r2_score(y_test, pred)
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].scatter(y_test, pred, s=8, alpha=.3); limits = [min(y_test.min(), pred.min()), max(y_test.max(), pred.max())]; ax[0].plot(limits, limits, "r--"); ax[0].set(xlabel="Actual ($100k)", ylabel="Predicted ($100k)", title="Actual vs predicted")
ax[1].scatter(pred, y_test - pred, s=8, alpha=.3); ax[1].axhline(0, color="red", ls="--"); ax[1].set(xlabel="Predicted ($100k)", ylabel="Residual ($100k)", title="Residuals")
fig.tight_layout(); fig.savefig(OUT / "regression_diagnostics.png", dpi=160); plt.close(fig)
report = f"Samples: {len(data.data)}\nTrain: {len(X_train)} | Test: {len(X_test)}\nMSE: {mse:.4f} ($100k units squared)\nRMSE: {rmse:.4f} ($100k units)\nMAE: {mae:.4f} ($100k units)\nR-squared: {r2:.4f}\n"
(OUT / "metrics.txt").write_text(report, encoding="utf-8")
print(report)
