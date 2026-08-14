"""
Save the TUNED XGBoost + SMOTE pipeline under NEW filenames (v2).
Original churn_pipeline.joblib / preprocessor.joblib / xgb_model.json
are NOT touched -- the currently deployed backend keeps working as-is.
"""
import json
import pandas as pd
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import classification_report, roc_auc_score, f1_score, recall_score

DATA_PATH = Path("telco_churn_clean.csv")
with open("best_xgb_params.json") as f:
    tuned = json.load(f)

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

full_pipeline = ImbPipeline(steps=[
    ("preprocessor", preprocessor),
    ("smote", SMOTE(random_state=42, **tuned["smote_params"])),
    ("model", XGBClassifier(random_state=42, eval_metric="logloss", **tuned["best_params"])),
])

full_pipeline.fit(X_train, y_train)

# Sanity check -- should match Step 2/3 numbers from tuning run
y_proba = full_pipeline.predict_proba(X_test)[:, 1]
threshold = tuned["optimal_threshold"]
y_pred = (y_proba >= threshold).astype(int)

print("=== v2 Pipeline (tuned XGBoost + SMOTE, optimal threshold) sanity check ===")
print(classification_report(y_test, y_pred))
print("Recall(churn):", round(recall_score(y_test, y_pred), 3))
print("F1(churn):", round(f1_score(y_test, y_pred), 3))
print("ROC-AUC:", round(roc_auc_score(y_test, y_proba), 4))

# --- Save under v2 names (originals left completely untouched) ---
joblib.dump(full_pipeline, "churn_pipeline_v2.joblib")

fitted_preprocessor = full_pipeline.named_steps["preprocessor"]
fitted_model = full_pipeline.named_steps["model"]

joblib.dump(fitted_preprocessor, "preprocessor_v2.joblib")
fitted_model.save_model("xgb_model_v2.json")

# Save threshold alongside so main_v2.py can load it
with open("model_config_v2.json", "w") as f:
    json.dump({"decision_threshold": threshold}, f, indent=2)

print("\nSaved: churn_pipeline_v2.joblib, preprocessor_v2.joblib, xgb_model_v2.json, model_config_v2.json")

# --- Schema check: confirm feature names match the ORIGINAL preprocessor ---
original_preprocessor = joblib.load("/mnt/user-data/uploads/preprocessor.joblib")
orig_features = list(original_preprocessor.get_feature_names_out())
new_features = list(fitted_preprocessor.get_feature_names_out())

print("\n=== Feature name schema check (v2 vs originally deployed) ===")
print("Same feature names, same order:", orig_features == new_features)
if orig_features != new_features:
    print("DIFF found -- original count:", len(orig_features), "new count:", len(new_features))
