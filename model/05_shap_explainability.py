"""
STAGE 5: SHAP Explainability

We load our saved XGBoost+SMOTE pipeline, then:
    1. GLOBAL explainability: which features matter most across ALL customers
    2. LOCAL explainability: for ONE specific customer, the top reasons behind
       their individual prediction

IMPORTANT DETAIL: SHAP needs to see the data AFTER preprocessing (numbers,
one-hot encoded), because that's what the model itself actually sees --
but we want to LABEL the SHAP plots with human-readable feature names
(e.g. "Contract_Month-to-month"), not just column indices. We get those
names from preprocessor.get_feature_names_out().
"""

import pandas as pd
from pathlib import Path
import joblib
import shap
import numpy as np

MODEL_PATH = Path(__file__).with_name("churn_pipeline.joblib")
DATA_PATH = Path(__file__).with_name("telco_churn_clean.csv")

# ---------------------------------------------------------------------
# Load the saved pipeline (preprocessor + SMOTE + XGBoost, all bundled)
# ---------------------------------------------------------------------
pipeline = joblib.load(MODEL_PATH)
preprocessor = pipeline.named_steps["preprocessor"]
model = pipeline.named_steps["model"]

df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["Churn"])
y = df["Churn"]

# Use the same train/test split as before so we explain UNSEEN test customers
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------------------------------------------
# Transform test data the same way the model sees it (numbers, one-hot)
# ---------------------------------------------------------------------
X_test_processed = preprocessor.transform(X_test)
feature_names = preprocessor.get_feature_names_out()

# Convert to a DataFrame purely so SHAP's plots/prints show real column names
X_test_df = pd.DataFrame(
    X_test_processed.toarray() if hasattr(X_test_processed, "toarray") else X_test_processed,
    columns=feature_names
)

# ---------------------------------------------------------------------
# STEP 1: Build the explainer and compute SHAP values for the whole test set
# TreeExplainer is fast + EXACT for tree-based models like XGBoost
# ---------------------------------------------------------------------
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test_df)

print("SHAP values shape:", shap_values.shape)   # (num_customers, num_features)
print("Base value (average model output before any features considered):",
      explainer.expected_value)

# ---------------------------------------------------------------------
# STEP 2: GLOBAL explainability -- average |impact| per feature across
# every customer in the test set. This tells us "overall, what matters most?"
# ---------------------------------------------------------------------
mean_abs_shap = np.abs(shap_values).mean(axis=0)
global_importance = pd.DataFrame({
    "feature": feature_names,
    "mean_abs_shap": mean_abs_shap
}).sort_values("mean_abs_shap", ascending=False)

print("\n=== TOP 10 Globally Important Features ===")
print(global_importance.head(10).to_string(index=False))

# ---------------------------------------------------------------------
# STEP 3: LOCAL explainability -- explain ONE specific customer's prediction
# ---------------------------------------------------------------------
def explain_customer(row_index, top_n=3):
    """Return the top_n features that most influenced this customer's prediction,
    with direction (pushed toward churn, or away from churn)."""
    customer_shap = shap_values[row_index]
    customer_data = X_test_df.iloc[row_index]

    prob = model.predict_proba(X_test_df.iloc[[row_index]])[0, 1]

    contributions = pd.DataFrame({
        "feature": feature_names,
        "shap_value": customer_shap,
        "feature_value": customer_data.values
    })
    contributions["abs_shap"] = contributions["shap_value"].abs()
    top_features = contributions.sort_values("abs_shap", ascending=False).head(top_n)

    print(f"\nCustomer #{row_index} -- Predicted churn probability: {prob:.2%}")
    for _, r in top_features.iterrows():
        direction = "INCREASES" if r["shap_value"] > 0 else "DECREASES"
        print(f"  - {r['feature']} {direction} churn risk "
              f"(SHAP={r['shap_value']:+.3f}, actual value={r['feature_value']:.2f})")

    return top_features


# Explain a couple of example customers
explain_customer(0)
explain_customer(5)
explain_customer(10)
