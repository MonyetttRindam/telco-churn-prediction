"""Pydantic schemas untuk request & response API."""

from pydantic import BaseModel, Field
from typing import Literal


class CustomerInput(BaseModel):
    """Data 1 pelanggan untuk prediksi churn."""
    
    # === DEMOGRAFIS ===
    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    
    # === LAYANAN ===
    tenure: int = Field(..., ge=0, le=100, description="Bulan langganan")
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    
    # === KONTRAK & PEMBAYARAN ===
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(..., ge=0, description="Tagihan bulanan ($)")
    TotalCharges: float = Field(..., ge=0, description="Total tagihan sampai sekarang ($)")


class PredictionOutput(BaseModel):
    """Hasil prediksi churn."""
    
    churn: int = Field(..., description="1 = churn, 0 = tidak churn")
    probability: float = Field(..., ge=0, le=1, description="Probabilitas churn (0-1)")
    threshold: float = Field(..., description="Threshold yang dipakai untuk decision")
    model_version: str = Field(..., description="Versi model yang membuat prediksi ini")