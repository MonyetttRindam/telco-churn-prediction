"""FastAPI app: Telco churn prediction + Phase 2 MLOps endpoints."""

from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException

from api.core.model_manager import ModelManager, ModelNotLoadedError
from api.deps import get_model_manager, init_dependencies
from api.routes import (
    batches_router,
    jobs_router,
    mlops_router,
    retrain_router,
)
from app.schemas import CustomerInput, PredictionOutput
from src.preprocessing import add_features, clean_raw

ROOT = Path(__file__).resolve().parent.parent

# FIXED decision threshold (MLOPS_PLAN.md: Critical Constraints). Matches
# models/model_config.json ("threshold": 0.6) and the retraining gate.
THRESHOLD = 0.60


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize all singletons (Registry/ModelManager/...) at startup."""
    init_dependencies(ROOT)
    yield
    # No shutdown cleanup: in-memory state, no persistent connections.


app = FastAPI(
    title="Telco Churn Prediction MLOps API",
    description="ML prediction + retraining pipeline with versioned model registry",
    version="2.0.0",
    lifespan=lifespan,
)

# Phase 2 routers (all under /api prefix).
app.include_router(mlops_router)
app.include_router(retrain_router)
app.include_router(batches_router)
app.include_router(jobs_router)


@app.get("/")
def root(model_manager: ModelManager = Depends(get_model_manager)):
    """Healthcheck + which model version is currently serving."""
    try:
        active = model_manager.get_current_version_id()
    except ModelNotLoadedError:
        active = None
    return {
        "status": "ok",
        "version": "2.0.0",
        "active_model": active,
        "threshold": THRESHOLD,
    }


@app.post("/predict", response_model=PredictionOutput)
def predict(
    customer: CustomerInput,
    model_manager: ModelManager = Depends(get_model_manager),
):
    """Prediksi churn untuk 1 pelanggan pakai model versi AKTIF.

    Model aktif di-swap in-memory oleh /api/retrain (promote) atau
    /api/rollback — tanpa restart server.
    """
    try:
        model, preprocessor, version_id = model_manager.get_current()
    except ModelNotLoadedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None

    # Pipeline preprocessing sama persis dgn training.
    df = pd.DataFrame([customer.model_dump()])
    df = clean_raw(df)
    df = add_features(df)
    X = preprocessor.transform(df)

    proba = float(model.predict_proba(X)[:, 1][0])
    churn = int(proba >= THRESHOLD)

    return PredictionOutput(
        churn=churn,
        probability=round(proba, 4),
        threshold=THRESHOLD,
        model_version=version_id,
    )
