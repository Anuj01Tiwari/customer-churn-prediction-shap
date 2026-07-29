import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def prepare_data():
    data_path = Path(__file__).with_name("telco_churn_clean.csv")
    df = pd.read_csv(data_path)

    X = df.drop(columns=["Churn"]).copy()
    y = df["Churn"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    numerical_cols = ["Tenure Months", "Monthly Charges", "Total Charges"]
    categorical_cols = [col for col in X_train.columns if col not in numerical_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
        ]
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    return X_train, X_test, y_train, y_test, X_train_processed, X_test_processed


if __name__ == "__main__":
    _, _, y_train, y_test, X_train_processed, X_test_processed = prepare_data()

    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    log_reg.fit(X_train_processed, y_train)
    y_pred_lr = log_reg.predict(X_test_processed)

    print("=== Logistic Regression ===")
    print(classification_report(y_test, y_pred_lr))
    print("ROC-AUC:", roc_auc_score(y_test, log_reg.predict_proba(X_test_processed)[:, 1]))

    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train_processed, y_train)
    y_pred_rf = rf.predict(X_test_processed)

    print("\n=== Random Forest ===")
    print(classification_report(y_test, y_pred_rf))
    print("ROC-AUC:", roc_auc_score(y_test, rf.predict_proba(X_test_processed)[:, 1]))

    xgb = XGBClassifier(n_estimators=200, random_state=42, eval_metric="logloss")
    xgb.fit(X_train_processed, y_train)
    y_pred_xgb = xgb.predict(X_test_processed)

    print("\n=== XGBoost ===")
    print(classification_report(y_test, y_pred_xgb))
    print("ROC-AUC:", roc_auc_score(y_test, xgb.predict_proba(X_test_processed)[:, 1]))