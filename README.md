# Customer Churn Prediction with SHAP Explainability

## 📌 Overview
An end-to-end Machine Learning project that predicts customer churn using a hyperparameter-tuned XGBoost model with SMOTE imbalance handling and explains every prediction using SHAP (SHapley Additive Explanations). The project includes a React frontend and FastAPI backend for interactive predictions.

---

## 🚀 Features
- Customer Churn Prediction
- Data Cleaning & Preprocessing
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Hyperparameter-Tuned XGBoost Classifier + SMOTE
- Optimal Decision Thresholding via Precision-Recall Curve Analysis
- SHAP Explainability
- React Frontend
- FastAPI Python Backend
- Portable Model Pipeline Serialization

---

## 🛠️ Tech Stack
- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Imbalanced-learn (SMOTE)
- SHAP
- FastAPI
- React
- Vite
- Joblib

---

## 📊 Dataset
IBM Telco Customer Churn Dataset

- 7,043 customer records
- 20+ customer behaviour features
- Binary classification (Churn / No Churn)

> [!NOTE]
> All data paths referenced in scripts (e.g. `01_data_cleaning_eda.py`) are relative to the `model/` directory (`telco_churn_clean.csv` and `Telco_customer_churn.xlsx`).

---

## 📈 Model Performance

Tuned XGBoost + SMOTE model performance on held-out test data with decision threshold optimized via Precision-Recall curve analysis:

- **Accuracy**: 82.3%
- **Precision (churn class)**: 65.1%
- **Recall (churn class)**: 80.0%
- **F1-score (churn class)**: 71.8%
- **ROC-AUC**: 0.88

### ⚙️ Tuning & Optimization Note
Hyperparameters were tuned via `RandomizedSearchCV` using 5-fold cross-validation (60 candidate configurations, optimizing F1-score). Rather than using the default 0.5 classification threshold, the decision threshold was optimized via precision-recall curve analysis to achieve high churn recall (80.0%) while maintaining strong precision.

---

## 📂 Project Structure

```
customer-churn-prediction-shap/
│── frontend/          # React + Vite UI
│── model/             # Data processing, training scripts, serialized model & backend
│   ├── Backend/       # Deployed FastAPI app and requirements
│   ├── main.py        # Live FastAPI backend application
│   ├── preprocessor.joblib # Live Scikit-learn ColumnTransformer
│   ├── xgb_model.json # Live XGBoost model file
│   ├── model_config.json # Live model config (decision threshold)
│   ├── churn_pipeline.joblib # Live full pipeline (preprocessor + SMOTE + XGBoost)
│   └── evaluate.py    # Pipeline evaluation script
│── README.md
```

---

## ▶️ How to Run

### Clone Repository

```bash
git clone https://github.com/Anuj01Tiwari/customer-churn-prediction-shap.git
```

### Install Dependencies

```bash
pip install -r model/Backend/requirements.txt
```

### Run Backend

```bash
cd model
uvicorn main:app --reload
```

---

## 🔍 Explainable AI

This project uses **SHAP (SHapley Additive Explanations)** to explain why the model predicts a customer is likely to churn by showing the contribution of each feature in plain language.

---

## 👨‍💻 Author

**Anuj Tiwari**