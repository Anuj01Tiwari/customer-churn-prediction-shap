import pandas as pd
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline   # supports SMOTE inside a Pipeline
from sklearn.metrics import classification_report, roc_auc_score

DATA_PATH = Path(__file__).with_name("telco_churn_clean.csv")
MODEL_PATH = Path(__file__).with_name("churn_pipeline.joblib")

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

# ImbPipeline (from imblearn) lets us chain: raw data -> preprocess -> SMOTE -> model
# A normal sklearn Pipeline can't include SMOTE because SMOTE changes the number
# of rows (adds synthetic ones) -- sklearn's own Pipeline doesn't support that,
# imblearn's version does.
full_pipeline = ImbPipeline(steps=[
    ("preprocessor", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("model", XGBClassifier(n_estimators=200, random_state=42, eval_metric="logloss")),
])

full_pipeline.fit(X_train, y_train)

# Final sanity check: confirm this saved pipeline reproduces our Stage 4 XGBoost+SMOTE numbers
y_pred = full_pipeline.predict(X_test)
y_proba = full_pipeline.predict_proba(X_test)[:, 1]
print("=== Final Pipeline (XGBoost + SMOTE) sanity check ===")
print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))

joblib.dump(full_pipeline, MODEL_PATH)
print(f"\nSaved final pipeline to: {MODEL_PATH}")
