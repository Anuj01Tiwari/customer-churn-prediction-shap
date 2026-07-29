"""
STAGE 6: FastAPI Backend

Exposes ONE main endpoint: POST /predict
    Input:  raw customer details (exactly what a form would submit)
    Output: churn probability + top 3 SHAP-based reasons in plain language

Run this with:
    uvicorn main:app --reload
Then open http://127.0.0.1:8000/docs in your browser to test it interactively.
"""

from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------
# STEP 1: Load the trained pipeline ONCE, when the server starts
# (NOT inside the endpoint function -- loading a model from disk is slow,
#  we only want to pay that cost once, not on every single request)
# ---------------------------------------------------------------------
from xgboost import XGBClassifier

# ---------------------------------------------------------------------
# STEP 1: Load the preprocessor and model SEPARATELY (portable approach)
#
# WHY separately, instead of one joblib.load() like before:
#   Pickling the raw XGBoost booster via joblib embeds an internal binary
#   buffer that can fail to load on a different OS/machine even at the
#   same XGBoost version ("input stream corrupted"). XGBoost's own
#   .json save format is explicitly built to be portable across
#   platforms, so we use that for the model, and joblib only for the
#   preprocessor (pure sklearn/numpy objects, always portable).
# ---------------------------------------------------------------------
PREPROCESSOR_PATH = Path(__file__).with_name("preprocessor.joblib")
MODEL_PATH = Path(__file__).with_name("xgb_model.json")

preprocessor = joblib.load(PREPROCESSOR_PATH)

model = XGBClassifier()
model.load_model(str(MODEL_PATH))

# Build the SHAP explainer once too -- reused across all requests
explainer = shap.TreeExplainer(model)
feature_names = preprocessor.get_feature_names_out()

# ---------------------------------------------------------------------
# Precompute a mapping from each ENCODED feature name (e.g. "cat__Contract_Two year")
# back to its ORIGINAL raw column name (e.g. "Contract"). We need this so we can
# show the customer's REAL input value in explanations, not the scaled/one-hot
# internal representation the model actually computes on.
# ---------------------------------------------------------------------
numerical_cols = list(preprocessor.transformers_[0][2])
categorical_cols = list(preprocessor.transformers_[1][2])
raw_columns_by_length = sorted(categorical_cols, key=len, reverse=True)


def map_to_raw_column(encoded_name: str) -> str:
    clean = encoded_name.replace("num__", "").replace("cat__", "")
    if clean in numerical_cols:
        return clean
    for raw_col in raw_columns_by_length:
        if clean.startswith(raw_col + "_"):
            return raw_col
    return clean


# ---------------------------------------------------------------------
# STEP 2: Define the request schema with Pydantic
#
# Field(alias=...) lets us accept JSON keys with spaces ("Tenure Months")
# while using valid Python variable names internally (tenure_months).
# ---------------------------------------------------------------------
class CustomerData(BaseModel):
    gender: Literal["Male", "Female"] = Field(alias="Gender")
    senior_citizen: Literal["Yes", "No"] = Field(alias="Senior Citizen")
    partner: Literal["Yes", "No"] = Field(alias="Partner")
    dependents: Literal["Yes", "No"] = Field(alias="Dependents")
    tenure_months: int = Field(alias="Tenure Months", ge=0, le=100)
    phone_service: Literal["Yes", "No"] = Field(alias="Phone Service")
    multiple_lines: Literal["Yes", "No", "No phone service"] = Field(alias="Multiple Lines")
    internet_service: Literal["DSL", "Fiber optic", "No"] = Field(alias="Internet Service")
    online_security: Literal["Yes", "No", "No internet service"] = Field(alias="Online Security")
    online_backup: Literal["Yes", "No", "No internet service"] = Field(alias="Online Backup")
    device_protection: Literal["Yes", "No", "No internet service"] = Field(alias="Device Protection")
    tech_support: Literal["Yes", "No", "No internet service"] = Field(alias="Tech Support")
    streaming_tv: Literal["Yes", "No", "No internet service"] = Field(alias="Streaming TV")
    streaming_movies: Literal["Yes", "No", "No internet service"] = Field(alias="Streaming Movies")
    contract: Literal["Month-to-month", "One year", "Two year"] = Field(alias="Contract")
    paperless_billing: Literal["Yes", "No"] = Field(alias="Paperless Billing")
    payment_method: Literal[
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ] = Field(alias="Payment Method")
    monthly_charges: float = Field(alias="Monthly Charges", ge=0)
    total_charges: float = Field(alias="Total Charges", ge=0)

    class Config:
        populate_by_name = True   # allows using either the alias or the python name


class Reason(BaseModel):
    feature: str
    direction: str          # "increases" or "decreases"
    description: str


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: str
    top_reasons: list[Reason]


# ---------------------------------------------------------------------
# STEP 3: Create the FastAPI app
# ---------------------------------------------------------------------
app = FastAPI(title="Customer Churn Prediction API")

# CORS: without this, a React app running on a different port (e.g. localhost:3000)
# would be BLOCKED by the browser from calling this API (localhost:8000) due to
# the browser's same-origin security policy. This explicitly allows it.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # for development; in production, list your real frontend URL
    allow_methods=["*"],
    allow_headers=["*"],
)


def humanize_reason(feature_name: str, shap_value: float, raw_input: dict) -> Reason:
    """Turn an encoded feature name + this customer's REAL input value into a
    readable sentence, e.g. 'Contract is Month-to-month, which increases churn risk'."""
    raw_col = map_to_raw_column(feature_name)
    raw_value = raw_input.get(raw_col, "N/A")
    direction = "increases" if shap_value > 0 else "decreases"
    description = f"{raw_col} is '{raw_value}', which {direction} this customer's churn risk"
    return Reason(feature=raw_col, direction=direction, description=description)


@app.get("/")
def health_check():
    return {"status": "Churn Prediction API is running"}


@app.post("/predict", response_model=PredictionResponse)
def predict_churn(customer: CustomerData):
    # STEP A: Convert the incoming request into a single-row DataFrame,
    # using the ORIGINAL column names (with spaces) the pipeline expects.
    input_dict = customer.model_dump(by_alias=True)
    input_df = pd.DataFrame([input_dict])

    # STEP B: Transform raw input, then get the real probability from the model
    input_processed = preprocessor.transform(input_df)
    churn_probability = float(model.predict_proba(input_processed)[0, 1])
    churn_prediction = "Yes" if churn_probability >= 0.5 else "No"
    input_processed_df = pd.DataFrame(
        input_processed.toarray() if hasattr(input_processed, "toarray") else input_processed,
        columns=feature_names
    )
    shap_values = explainer.shap_values(input_processed_df)[0]

    # STEP D: Pick the top 3 features by absolute SHAP impact
    top_indices = np.argsort(np.abs(shap_values))[::-1][:3]
    top_reasons = [
        humanize_reason(feature_names[i], shap_values[i], input_dict) for i in top_indices
    ]

    return PredictionResponse(
        churn_probability=round(churn_probability, 4),
        churn_prediction=churn_prediction,
        top_reasons=top_reasons,
    )
