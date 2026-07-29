import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, f1_score, recall_score
from imblearn.over_sampling import SMOTE


def prepare_data():
    data_path = Path(__file__).with_name("telco_churn_clean.csv")
    df = pd.read_csv(data_path)

    X = df.drop(columns=["Churn"]).copy()
    y = df["Churn"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    numerical_cols = ["Tenure Months", "Monthly Charges", "Total Charges"]
    categorical_cols = [c for c in X_train.columns if c not in numerical_cols]

    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
    ])

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    return X_train_processed, X_test_processed, y_train, y_test, preprocessor


def evaluate(name, model, X_test, y_test, results):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    recall_1 = recall_score(y_test, y_pred, pos_label=1)
    f1_1 = f1_score(y_test, y_pred, pos_label=1)
    auc = roc_auc_score(y_test, y_proba)

    print(f"\n=== {name} ===")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC: {auc:.4f}")

    results.append({
        "Model": name, "Recall(churn)": round(recall_1, 3),
        "F1(churn)": round(f1_1, 3), "ROC-AUC": round(auc, 4)
    })


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, preprocessor = prepare_data()
    results = []

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=200, random_state=42, eval_metric="logloss"),
    }

    # ---------------------------------------------------------------
    # STRATEGY 1: Baseline (no imbalance handling) -- from Stage 3
    # ---------------------------------------------------------------
    for name, model in models.items():
        model.fit(X_train, y_train)
        evaluate(f"{name} (Baseline)", model, X_test, y_test, results)

    # ---------------------------------------------------------------
    # STRATEGY 2: class_weight='balanced'
    # ---------------------------------------------------------------
    weighted_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"),
        # XGBoost uses scale_pos_weight instead of class_weight -- ratio of negative to positive class
        "XGBoost": XGBClassifier(
            n_estimators=200, random_state=42, eval_metric="logloss",
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum()
        ),
    }
    for name, model in weighted_models.items():
        model.fit(X_train, y_train)
        evaluate(f"{name} (class_weight balanced)", model, X_test, y_test, results)

    # ---------------------------------------------------------------
    # STRATEGY 3: SMOTE (applied ONLY to training data)
    # ---------------------------------------------------------------
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    print(f"\nBefore SMOTE: {y_train.value_counts().to_dict()}")
    print(f"After SMOTE:  {y_train_smote.value_counts().to_dict()}")

    smote_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=200, random_state=42, eval_metric="logloss"),
    }
    for name, model in smote_models.items():
        model.fit(X_train_smote, y_train_smote)
        evaluate(f"{name} (SMOTE)", model, X_test, y_test, results)

    # ---------------------------------------------------------------
    # Final comparison table
    # ---------------------------------------------------------------
    results_df = pd.DataFrame(results).sort_values("F1(churn)", ascending=False)
    print("\n\n===== FINAL COMPARISON (sorted by F1 on churn class) =====")
    print(results_df.to_string(index=False))