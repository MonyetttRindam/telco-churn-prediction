"""FastAPI app untuk prediksi churn Telco."""

import joblib
import json
import pandas as pd
from pathlib import Path
from fastapi import FastAPI
from app.schemas import CustomerInput, PredictionOutput
from src.preprocessing import clean_raw, add_features

# === Setup paths ===
ROOT = Path(__file__).parent.parent

# === Load artifacts (sekali aja saat startup) ===
preprocessor = joblib.load(ROOT / "models/preprocessor.joblib")
model = joblib.load(ROOT / "models/model_final.joblib")
with open(ROOT / "models/model_config.json") as f:
    config = json.load(f)

THRESHOLD = config["threshold"]

# === FastAPI app ===
app = FastAPI(
    title="Telco Churn Prediction API",
    description="Prediksi churn pelanggan Telco",
    version="1.0",
)


@app.get("/")
def root():
    """Healthcheck."""
    return {
        "status": "ok",
        "model": config["model_type"],
        "threshold": THRESHOLD,
    }


@app.post("/predict", response_model=PredictionOutput)
def predict(customer: CustomerInput):
    """Prediksi churn untuk 1 pelanggan."""
    
    # 1. Convert Pydantic ke DataFrame
    df = pd.DataFrame([customer.model_dump()])
    
    # 2. Pipeline preprocessing (sama persis dgn test_load.py)
    df = clean_raw(df)
    df = add_features(df)
    
    # 3. Transform pakai preprocessor (stateful, dari joblib)
    X = preprocessor.transform(df)
    
    # 4. Predict probabilitas
    proba = float(model.predict_proba(X)[:, 1][0])
    
    # 5. Apply threshold
    churn = int(proba >= THRESHOLD)
    
    # 6. Return sesuai schema PredictionOutput
    return PredictionOutput(
        churn=churn,
        probability=round(proba, 4),
        threshold=THRESHOLD,
    )