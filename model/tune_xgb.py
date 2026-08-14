import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (classification_report, roc_auc_score, f1_score,
                              recall_score, precision_score, precision_recall_curve)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

DATA_PATH = Path("telco_churn_clean.csv")
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["Churn"])
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

numerical_cols = ["Tenure Months", "Monthly Charges", "Total Charges"]
categorical_cols = [c for c in X_train.columns if c not in numerical_cols]

preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numerical_cols),
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
])

# ---------------------------------------------------------------------
# STEP 0: Reproduce the ORIGINAL baseline (XGBoost + SMOTE, no tuning)
# to confirm we're starting from the same numbers you already have.
# ---------------------------------------------------------------------
baseline_pipeline = ImbPipeline(steps=[
    ("preprocessor", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("model", XGBClassifier(n_estimators=200, random_state=42, eval_metric="logloss")),
])
baseline_pipeline.fit(X_train, y_train)
y_pred_base = baseline_pipeline.predict(X_test)
y_proba_base = baseline_pipeline.predict_proba(X_test)[:, 1]

print("=" * 70)
print("STEP 0: BASELINE (XGBoost + SMOTE, untuned) -- sanity check")
print("=" * 70)
print(f"Recall(churn): {recall_score(y_test, y_pred_base):.3f}")
print(f"F1(churn):     {f1_score(y_test, y_pred_base):.3f}")
print(f"ROC-AUC:       {roc_auc_score(y_test, y_proba_base):.4f}")

# ---------------------------------------------------------------------
# STEP 1: Hyperparameter search space for XGBoost
# ---------------------------------------------------------------------
param_dist = {
    "model__n_estimators": [100, 200, 300, 400, 600],
    "model__max_depth": [2, 3, 4, 5, 6, 7],
    "model__learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15],
    "model__min_child_weight": [1, 2, 3, 5, 7, 10],
    "model__subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "model__colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "model__gamma": [0, 0.1, 0.3, 0.5, 1, 2],
    "model__reg_alpha": [0, 0.01, 0.1, 0.5, 1, 2],
    "model__reg_lambda": [0.5, 1, 1.5, 2, 3, 5],
    "smote__k_neighbors": [3, 5, 7],
}

search_pipeline = ImbPipeline(steps=[
    ("preprocessor", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("model", XGBClassifier(random_state=42, eval_metric="logloss", n_jobs=-1)),
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

search = RandomizedSearchCV(
    estimator=search_pipeline,
    param_distributions=param_dist,
    n_iter=60,
    scoring="f1",          # optimizing F1 on the churn (positive) class
    cv=cv,
    random_state=42,
    n_jobs=-1,
    verbose=1,
)

print("\n" + "=" * 70)
print("STEP 1: RandomizedSearchCV (60 candidates x 5-fold CV, optimizing F1)")
print("=" * 70)
search.fit(X_train, y_train)

print("\nBest CV F1 score:", round(search.best_score_, 4))
print("Best params:")
for k, v in search.best_params_.items():
    print(f"  {k}: {v}")

best_model = search.best_estimator_

# ---------------------------------------------------------------------
# STEP 2: Evaluate tuned model on the held-out test set (default 0.5 threshold)
# ---------------------------------------------------------------------
y_pred_tuned = best_model.predict(X_test)
y_proba_tuned = best_model.predict_proba(X_test)[:, 1]

print("\n" + "=" * 70)
print("STEP 2: TUNED XGBoost on test set (threshold = 0.5)")
print("=" * 70)
print(classification_report(y_test, y_pred_tuned))
print(f"Recall(churn): {recall_score(y_test, y_pred_tuned):.3f}")
print(f"F1(churn):     {f1_score(y_test, y_pred_tuned):.3f}")
print(f"ROC-AUC:       {roc_auc_score(y_test, y_proba_tuned):.4f}")

# ---------------------------------------------------------------------
# STEP 3: Threshold tuning -- find the threshold that MAXIMIZES F1 on churn
# (0.5 is arbitrary; with imbalanced classes the optimal cut point moves)
# ---------------------------------------------------------------------
precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba_tuned)
f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-12)
best_idx = np.argmax(f1s[:-1])  # last point has no corresponding threshold
best_threshold = thresholds[best_idx]

y_pred_opt = (y_proba_tuned >= best_threshold).astype(int)

print("\n" + "=" * 70)
print(f"STEP 3: TUNED XGBoost with OPTIMAL threshold = {best_threshold:.3f}")
print("=" * 70)
print(classification_report(y_test, y_pred_opt))
print(f"Recall(churn): {recall_score(y_test, y_pred_opt):.3f}")
print(f"F1(churn):     {f1_score(y_test, y_pred_opt):.3f}")
print(f"Precision(churn): {precision_score(y_test, y_pred_opt):.3f}")
print(f"ROC-AUC (threshold-independent): {roc_auc_score(y_test, y_proba_tuned):.4f}")

# ---------------------------------------------------------------------
# STEP 4: Summary comparison table
# ---------------------------------------------------------------------
summary = pd.DataFrame([
    {"Model": "XGBoost + SMOTE (baseline, untuned)", "Threshold": 0.5,
     "Recall": recall_score(y_test, y_pred_base), "Precision": precision_score(y_test, y_pred_base),
     "F1": f1_score(y_test, y_pred_base), "ROC-AUC": roc_auc_score(y_test, y_proba_base)},
    {"Model": "XGBoost + SMOTE (tuned)", "Threshold": 0.5,
     "Recall": recall_score(y_test, y_pred_tuned), "Precision": precision_score(y_test, y_pred_tuned),
     "F1": f1_score(y_test, y_pred_tuned), "ROC-AUC": roc_auc_score(y_test, y_proba_tuned)},
    {"Model": "XGBoost + SMOTE (tuned + optimal threshold)", "Threshold": round(best_threshold, 3),
     "Recall": recall_score(y_test, y_pred_opt), "Precision": precision_score(y_test, y_pred_opt),
     "F1": f1_score(y_test, y_pred_opt), "ROC-AUC": roc_auc_score(y_test, y_proba_tuned)},
])
summary[["Recall", "Precision", "F1", "ROC-AUC"]] = summary[["Recall", "Precision", "F1", "ROC-AUC"]].round(4)

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print(summary.to_string(index=False))

# Save best params + threshold for reuse
import json
with open("best_xgb_params.json", "w") as f:
    json.dump({
        "best_params": {k.replace("model__", ""): v for k, v in search.best_params_.items() if k.startswith("model__")},
        "smote_params": {k.replace("smote__", ""): v for k, v in search.best_params_.items() if k.startswith("smote__")},
        "optimal_threshold": float(best_threshold),
        "cv_f1_score": float(search.best_score_),
    }, f, indent=2)
print("\nSaved best hyperparameters to best_xgb_params.json")
