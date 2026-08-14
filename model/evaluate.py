"""
Evaluate the TUNED XGBoost + SMOTE pipeline (churn_pipeline_v2.joblib)
Run this from inside your `model` folder:

    python evaluate_v2.py

Prints accuracy, precision, recall, F1, confusion matrix, ROC-AUC in the terminal.
"""
import json
import pandas as pd
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score
)

DATA_PATH = Path(__file__).with_name("telco_churn_clean.csv")
PIPELINE_PATH = Path(__file__).with_name("churn_pipeline.joblib")
CONFIG_PATH = Path(__file__).with_name("model_config.json")

# Load the exact same train/test split used during training (random_state=42)
# so we're evaluating on the same held-out test set.
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["Churn"])
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipeline = joblib.load(PIPELINE_PATH)

with open(CONFIG_PATH) as f:
    threshold = json.load(f)["decision_threshold"]

y_proba = pipeline.predict_proba(X_test)[:, 1]
y_pred = (y_proba >= threshold).astype(int)

print("=" * 60)
print(f"TUNED XGBoost + SMOTE  |  decision threshold = {threshold:.3f}")
print("=" * 60)

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=["No Churn (0)", "Churn (1)"]))

print("--- Confusion Matrix ---")
cm = confusion_matrix(y_test, y_pred)
print(f"                 Predicted No   Predicted Yes")
print(f"Actual No        {cm[0][0]:<14} {cm[0][1]}")
print(f"Actual Yes       {cm[1][0]:<14} {cm[1][1]}")

print("\n--- Summary Metrics ---")
print(f"Accuracy:            {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision (churn):   {precision_score(y_test, y_pred):.4f}")
print(f"Recall (churn):      {recall_score(y_test, y_pred):.4f}")
print(f"F1-score (churn):    {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC:             {roc_auc_score(y_test, y_proba):.4f}")
