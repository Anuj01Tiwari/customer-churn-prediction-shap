import pandas as pd
from pathlib import Path
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

    print("Train shape:", X_train_processed.shape)
    print("Test shape:", X_test_processed.shape)

    return X_train, X_test, y_train, y_test, X_train_processed, X_test_processed


if __name__ == "__main__":
    prepare_data()