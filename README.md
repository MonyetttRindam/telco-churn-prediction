# 📊 Telco Customer Churn Prediction

> End-to-end Machine Learning project — from EDA to a deployed, containerized prediction service — that identifies telecom customers at risk of *churn* (leaving the service) so retention teams can intervene before they go.

[![Live Streamlit](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit-production-0be4.up.railway.app)
[![Live FastAPI](https://img.shields.io/badge/Live%20API-Swagger%20Docs-009688?logo=fastapi&logoColor=white)](https://telco-churn-prediction-production.up.railway.app/docs)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)](#-docker-setup)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 🌐 Live Demo

The full system is deployed on **Railway** as two independent services that talk to each other over HTTP:

| Service | URL | What it is |
|---|---|---|
| 🎨 **Streamlit UI** | **https://streamlit-production-0be4.up.railway.app** | Interactive form — fill in a customer profile, get a churn prediction + business interpretation |
| ⚡ **FastAPI (Swagger)** | **https://telco-churn-prediction-production.up.railway.app/docs** | Interactive API docs — try `POST /predict` straight from the browser |

> **Heads up:** the live demo runs on Railway's free trial (~30 days), so the URLs may go offline after that window. The project is **fully reproducible locally** in one command — `docker compose up` — so you can always spin up an identical environment yourself (see [Docker Setup](#-docker-setup)).

> First request after an idle period may take a few seconds while the service wakes up — that's expected on the free tier.

---

## 🎯 Problem Statement

A telecom company is losing **26.5%** of its customers (1,869 out of 7,043). Assuming an ARPU of Rp 500k/month, that churn represents a potential loss of **~Rp 934 million/month**.

The goal of this project is to **predict which customers will churn** *before* they leave, so the retention team can act with targeted interventions (promos, follow-ups) instead of guessing.

The core challenge is that the dataset is **imbalanced** (only 26.5% positive), which makes *accuracy* misleading. The metrics that matter here are **Recall** and **F1** — catching as many would-be churners as possible.

---

## 📈 Key Results

Three models were compared on the test set (default threshold 0.5):

| Model | Recall | F1 | ROC-AUC | Accuracy |
|---|---|---|---|---|
| **Logistic Regression** ✅ | 0.791 | 0.612 | **0.842** | 0.733 |
| Random Forest | 0.757 | **0.634** | 0.842 | 0.768 |
| XGBoost | 0.668 | 0.597 | 0.818 | 0.761 |

**Final model: Logistic Regression** — chosen for the **highest recall & ROC-AUC**, plus it's *interpretable* and cheap to deploy (even though Random Forest's F1 is marginally higher).

After **threshold tuning to 0.60** (maximizing F1):

| Metric | Score |
|---|---|
| Precision | 0.561 |
| Recall | 0.717 |
| F1 | **0.629** |
| ROC-AUC | 0.842 |

> 💰 **Estimated business impact:** ~**Rp 203 million/year** net profit from targeted retention (assuming ARPU Rp 500k, promo cost Rp 50k, retention rate 60%). See [docs/MODEL_CARD.md](docs/MODEL_CARD.md) for the full breakdown and assumptions.

---

## 🏗️ Architecture

```
┌─────────────┐     ┌────────────────────────┐     ┌──────────────────┐
│  Raw Data   │────▶│  Preprocessing          │────▶│  Model Training  │
│ (IBM Telco) │     │  src/preprocessing.py   │     │  notebooks/  +   │
└─────────────┘     └────────────────────────┘     │  MLflow tracking │
                                                     └────────┬─────────┘
                                                              │ artifacts
                                                              ▼
                            ┌──────────────────────────────────────────┐
                            │ models/  (preprocessor · model · config)  │
                            └────────────────┬─────────────────────────┘
                                             │
                        ┌────────────────────┴────────────────────┐
                        ▼                                          ▼
                ┌────────────────┐                       ┌──────────────────┐
                │   FastAPI      │◀──────HTTP / JSON──────│   Streamlit UI   │
                │   app/main.py  │                        │ streamlit_app.py │
                │   POST /predict│                        └──────────────────┘
                └────────────────┘
```

The preprocessing logic (`src/preprocessing.py`) is used **identically** during training and inside the API — no duplicated transform that could silently drift between train and serve.

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| **ML & Data** | scikit-learn, pandas, XGBoost |
| **Experiment Tracking** | MLflow (SQLite backend) |
| **Hyperparameter Tuning** | Optuna (TPE) |
| **API Serving** | FastAPI + Uvicorn + Pydantic |
| **Frontend** | Streamlit |
| **EDA / Viz** | matplotlib, seaborn |
| **Containerization** | Docker + Docker Compose |
| **Deployment** | Railway (2 services / 1 project) |

---

## 🚀 Quick Start

Pick whichever path fits you:

### Option 1 — 🌐 Try the Live Demo (zero setup)

Just open the hosted apps — nothing to install:

- **Streamlit UI:** https://streamlit-production-0be4.up.railway.app
- **API Swagger:** https://telco-churn-prediction-production.up.railway.app/docs

### Option 2 — 🐳 Run with Docker (recommended for local)

The fastest way to run the **exact same setup** as production — both services, one command:

```bash
git clone https://github.com/MonyetttRindam/telco-churn-prediction.git
cd telco-churn-prediction
docker compose up --build
```

Then open:

- Streamlit → http://localhost:8501
- FastAPI docs → http://localhost:8000/docs

Full details, troubleshooting, and env vars are in [Docker Setup](#-docker-setup).

### Option 3 — 🐍 Run with local Python

```bash
# 1. Clone & set up environment
git clone https://github.com/MonyetttRindam/telco-churn-prediction.git
cd telco-churn-prediction
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)

# 2. Install dependencies
pip install -r requirements.txt         # production (API + Streamlit)
# pip install -r requirements-dev.txt   # + training / EDA / notebooks

# 3. Run the API
uvicorn app.main:app --reload           # -> http://127.0.0.1:8000/docs

# 4. Run Streamlit (separate terminal)
streamlit run streamlit_app.py          # -> http://localhost:8501

# 5. (optional) Inspect experiments in MLflow
mlflow ui --backend-store-uri sqlite:///mlflow.db   # -> http://localhost:5000
```

> The dataset (`data/raw/Telco-Customer-Churn.csv`) is **not** bundled in the repo. Download it from [Kaggle — IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and place it in `data/raw/` if you want to re-run the notebooks. The trained model artifacts **are** committed, so the API and Streamlit run without the raw data.

---

## 🐳 Docker Setup

The project ships **two Dockerfiles** — `Dockerfile.api` and `Dockerfile.streamlit` — orchestrated by `docker-compose.yml`. Both build on `python:3.12-slim`, install only `requirements.txt` (production deps), and expose healthchecks.

### Prerequisites

- [Docker Engine](https://docs.docker.com/get-docker/) 20.10+
- [Docker Compose](https://docs.docker.com/compose/) v2 (bundled with modern Docker Desktop — use `docker compose`, not `docker-compose`)

### Build & run

```bash
# Build both images and start the stack (foreground, logs streamed)
docker compose up --build

# Or run detached (in the background)
docker compose up --build -d
```

What happens:

1. **`api`** service builds from `Dockerfile.api`, serves FastAPI on port **8000**.
2. **`streamlit`** service builds from `Dockerfile.streamlit`, serves the UI on port **8501**.
3. Compose waits for the API healthcheck to pass (`depends_on: condition: service_healthy`) before Streamlit starts, so the UI never boots against a dead API.

Open:

| App | Local URL |
|---|---|
| Streamlit UI | http://localhost:8501 |
| FastAPI docs | http://localhost:8000/docs |
| API health | http://localhost:8000/ |

### Stop & clean up

```bash
docker compose down              # stop and remove containers + network
docker compose down --rmi local  # also remove the built images
docker compose logs -f api       # tail logs for a single service
docker compose ps                # see status + healthcheck state
```

### Environment variables

| Variable | Service | Default | Purpose |
|---|---|---|---|
| `API_URL` | streamlit | `http://localhost:8000` | Where the Streamlit app sends `POST /predict`. In Compose it's set to `http://api:8000` (Docker's internal DNS resolves `api` to the backend container). |
| `PORT` | api / streamlit | `8000` / `8501` | The port each service binds to. Falls back to the defaults via `${PORT:-8000}` / `${PORT:-8501}` — this is what makes the same image work locally **and** on Railway (which injects its own `PORT`). |

> **Why `API_URL` matters:** inside Compose the two containers share a bridge network (`telco-net`), so Streamlit reaches the API at `http://api:8000` — no public URL needed. In production on Railway (where the services don't share a private network by default) it points to the API's public URL instead. Same code, just a different `API_URL`.

### Troubleshooting

| Symptom | Likely cause & fix |
|---|---|
| `Streamlit: API tidak terhubung` | API container isn't healthy yet. Check `docker compose ps` — wait for `api` to report `healthy`, or inspect `docker compose logs api`. |
| `port is already allocated` | Something else is using 8000/8501. Stop it, or remap in `docker-compose.yml` (e.g. `"8080:8000"`). |
| Build fails on `pip install` | Stale layer cache. Rebuild clean: `docker compose build --no-cache`. |
| Changes not reflected | Images are built, not mounted. Rerun with `--build` after editing code. |
| Healthcheck stuck `starting` | The API loads model artifacts at startup; give it the `start_period` (10s) before it flips to `healthy`. |

---

## ☁️ Deployment (Railway)

The live demo is deployed to **[Railway](https://railway.app/)** as **two services within a single project**, mirroring the Compose setup:

```
Railway Project: telco-churn-prediction
├── Service 1: api         (Dockerfile.api)        → public URL
└── Service 2: streamlit   (Dockerfile.streamlit)  → public URL
```

### How it works

- **Two services, one project.** Each service points at its own Dockerfile (`Dockerfile.api` / `Dockerfile.streamlit`) from the same repo. Railway builds and deploys them independently.
- **Dynamic `PORT`.** Railway assigns a random port at runtime and injects it as the `PORT` env var. The containers honor it via `${PORT:-8000}` / `${PORT:-8501}` in their `CMD`, so the **same image runs unchanged** locally (default port) and on Railway (injected port). No hardcoded ports.
- **Inter-service communication.** Unlike Compose's private `telco-net` bridge, the two Railway services reach each other over their **public URLs**. The Streamlit service sets `API_URL` to the API's public URL:

  ```
  API_URL = https://telco-churn-prediction-production.up.railway.app
  ```

  Because the Streamlit code reads the endpoint from `os.getenv("API_URL", ...)`, switching from local → Compose → Railway is purely a config change — no code edits.

### Deploy it yourself (sketch)

```bash
# 1. Push the repo to GitHub
# 2. In Railway: New Project → Deploy from GitHub repo
# 3. Add Service #1 (api):
#      - Root: this repo, Dockerfile path: Dockerfile.api
#      - Generate a public domain
# 4. Add Service #2 (streamlit):
#      - Same repo, Dockerfile path: Dockerfile.streamlit
#      - Set env var:  API_URL = <public URL of the api service>
#      - Generate a public domain
# 5. Both services auto-redeploy on every push to the default branch.
```

---

## 📁 Project Structure

```
telco-churn-prediction/
├── app/                      # FastAPI service
│   ├── main.py               #   endpoints: GET / , POST /predict
│   └── schemas.py            #   Pydantic request/response models
├── src/
│   └── preprocessing.py      # reusable cleaning + feature engineering
├── notebooks/
│   ├── 01_EDA.ipynb          # exploratory data analysis
│   ├── 02_preprocessing.ipynb# pipeline + MLflow intro
│   └── 03_Modelling.ipynb    # modeling, tuning, threshold, business impact
├── models/                   # trained artifacts (committed)
│   ├── model_final.joblib
│   ├── preprocessor.joblib
│   └── model_config.json
├── scripts/
│   └── smoke_test.py         # demo preprocessing + MLflow experiment
├── figures/                  # EDA & evaluation plots
├── docs/
│   ├── MODEL_CARD.md         # model card (HF style)
│   ├── API_USAGE.md          # request/response examples
│   └── screenshots/          # demo images
├── data/raw/                 # dataset (not committed)
├── streamlit_app.py          # Streamlit frontend
├── Dockerfile.api            # FastAPI image
├── Dockerfile.streamlit      # Streamlit image
├── docker-compose.yml        # 2-service local orchestration
├── requirements.txt          # production deps
├── requirements-dev.txt      # + training / EDA deps
└── README.md
```

---

## 🔬 ML Pipeline

1. **EDA** ([01_EDA.ipynb](notebooks/01_EDA.ipynb)) — distributions, churn rate per feature, correlations.
2. **Preprocessing** ([02_preprocessing.ipynb](notebooks/02_preprocessing.ipynb)) — cleaning `TotalCharges`, 4 engineered features (`num_addons`, `is_new_customer`, `has_internet`, `tenure_group`), a `ColumnTransformer` (scale + one-hot) → 52 features.
3. **Modeling** ([03_Modelling.ipynb](notebooks/03_Modelling.ipynb)) — 3 baseline models, Optuna tuning, **threshold tuning (0.60)**, business impact, all tracked in MLflow.

Full model details → [docs/MODEL_CARD.md](docs/MODEL_CARD.md). API usage → [docs/API_USAGE.md](docs/API_USAGE.md).

---

## 🔮 Future Improvements

- [ ] **CI/CD** via GitHub Actions — run tests + auto-deploy to Railway on every push.
- [ ] **Model monitoring & drift detection** with [Evidently AI](https://www.evidentlyai.com/) — track data/prediction drift in production.
- [ ] **A/B testing** — serve two model/threshold variants and compare retention lift on real traffic.
- [ ] **Migrate to GCP Cloud Run** — container-native autoscaling beyond the Railway free tier.
- [ ] **Comprehensive `pytest` suite** — unit tests for preprocessing, schema validation, and API contract tests.

---

## 👤 Author

**Muhammad Abil Khoiri** — Computer Engineering, Telkom University

Data Science / ML Engineering portfolio project.

- GitHub: [@MonyetttRindam](https://github.com/MonyetttRindam)
- LinkedIn: [Muhammad Abil Khoiri](https://www.linkedin.com/in/muhammadabilkhoirii/)

---

## 📄 License

MIT License — see [LICENSE](LICENSE).
