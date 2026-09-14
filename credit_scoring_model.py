import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
    classification_report, RocCurveDisplay
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "credit_scoring_dataset.csv")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

features = [
    "age", "annual_income", "total_debt", "credit_history_years",
    "payment_history_score", "late_payments", "existing_loans",
    "employment_years", "credit_utilization", "debt_to_income"
]
target = "creditworthy"

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=2000, random_state=42))
])

model.fit(X_train, y_train)
pred = model.predict(X_test)
prob = model.predict_proba(X_test)[:, 1]

metrics = {
    "Accuracy": accuracy_score(y_test, pred),
    "Precision": precision_score(y_test, pred),
    "Recall": recall_score(y_test, pred),
    "F1-Score": f1_score(y_test, pred),
    "ROC-AUC": roc_auc_score(y_test, prob)
}

print("\n===== CREDIT SCORING MODEL =====")
for name, value in metrics.items():
    print(f"{name}: {value:.4f}")

print("\nClassification Report:\n")
print(classification_report(y_test, pred, target_names=["Not Creditworthy", "Creditworthy"]))

with open(os.path.join(OUT_DIR, "metrics.txt"), "w") as f:
    for name, value in metrics.items():
        f.write(f"{name}: {value:.4f}\n")

cm = confusion_matrix(y_test, pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["Not Creditworthy", "Creditworthy"])
disp.plot()
plt.title("Credit Scoring - Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"), dpi=160)
plt.close()

RocCurveDisplay.from_predictions(y_test, prob)
plt.title("Credit Scoring - ROC Curve")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "roc_curve.png"), dpi=160)
plt.close()

joblib.dump(model, os.path.join(MODEL_DIR, "credit_scoring_model.joblib"))

# Example prediction
sample = pd.DataFrame([{
    "age": 30,
    "annual_income": 72000,
    "total_debt": 18000,
    "credit_history_years": 8,
    "payment_history_score": 91,
    "late_payments": 0,
    "existing_loans": 1,
    "employment_years": 6,
    "credit_utilization": 0.22,
    "debt_to_income": 0.25
}])
sample_pred = model.predict(sample)[0]
sample_prob = model.predict_proba(sample)[0, 1]
print(f"\nExample applicant prediction: {'Creditworthy' if sample_pred == 1 else 'Not Creditworthy'}")
print(f"Creditworthy probability: {sample_prob:.2%}")