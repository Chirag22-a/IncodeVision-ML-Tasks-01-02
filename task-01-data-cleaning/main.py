from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "reports"
OUT.mkdir(exist_ok=True)
df = pd.read_csv(ROOT / "data/titanic.csv")
before = df.shape
missing = df.isna().sum()
duplicates = int(df.duplicated().sum())
df = df.drop_duplicates().copy()
for col in df.select_dtypes(include="number"):
    df[col] = df[col].fillna(df[col].median())
for col in df.select_dtypes(exclude="number"):
    if col == "deck":
        # Most deck values are absent in this teaching dataset; mode-imputing
        # would invent a deck for those passengers, so preserve the uncertainty.
        df[col] = df[col].fillna("Unknown")
    else:
        mode = df[col].mode()
        df[col] = df[col].fillna(mode.iloc[0] if not mode.empty else "Unknown")
for col in ("survived", "pclass", "sibsp", "parch"):
    if col in df:
        df[col] = df[col].astype("int64")
flags = {}
for col in df.select_dtypes(include="number"):
    q1, q3 = df[col].quantile([.25, .75])
    low, high = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    flags[col] = int(((df[col] < low) | (df[col] > high)).sum())
df.to_csv(OUT / "titanic_cleaned.csv", index=False)
fig, ax = plt.subplots(2, 2, figsize=(12, 8))
sns.histplot(df["age"], kde=True, ax=ax[0, 0]); ax[0, 0].set_title("Passenger age")
sns.countplot(data=df, x="class", hue="survived", ax=ax[0, 1]); ax[0, 1].set_title("Survival by class")
sns.boxplot(data=df, x="survived", y="fare", ax=ax[1, 0]); ax[1, 0].set_title("Fare by survival")
sns.heatmap(df.select_dtypes(include="number").corr(), annot=True, cmap="vlag", center=0, ax=ax[1, 1]); ax[1, 1].set_title("Numeric correlation")
fig.tight_layout(); fig.savefig(OUT / "eda.png", dpi=160); plt.close(fig)
report = [f"Rows/columns before: {before}", f"Rows/columns after: {df.shape}", f"Exact duplicate rows removed: {duplicates}", "Missing values before cleaning:", missing.to_string(), "Missing values after cleaning:", df.isna().sum().to_string(), "IQR outlier counts (flagged, retained):", str(flags), "Note: exact matching rows were removed; this dataset has no passenger ID to confirm identity.", "Note: missing deck values were labeled Unknown because most deck entries are absent."]
(OUT / "summary.txt").write_text("\n".join(report), encoding="utf-8")
print(f"Task 01 complete. See {OUT}")
