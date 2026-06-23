import joblib
import json
import pandas as pd
from pathlib import Path
from src.preprocessing import clean_raw, add_features

ROOT = Path(__file__).parent

preprocessor = joblib.load(ROOT / "models/preprocessor.joblib")
model = joblib.load(ROOT / "models/model_final.joblib")
with open(ROOT / "models/model_config.json") as f:
    config = json.load(f)

print(f"Model     : {config['model_type']}")
print(f"Threshold : {config['threshold']}")

df = pd.read_csv(ROOT / "data/raw/Telco-Customer-Churn.csv")
sample = df.head(1)
sample = clean_raw(sample)
sample = add_features(sample)
sample = sample.drop(columns=["Churn"], errors="ignore")

X = preprocessor.transform(sample)
proba = model.predict_proba(X)[:, 1][0]
pred = int(proba >= config["threshold"])

print(f"Proba     : {proba:.4f}")
print(f"Pred      : {'CHURN' if pred else 'NO CHURN'}")