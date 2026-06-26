# MLOps Retraining System - Implementation Plan

## Decisions

### Design decisions (Phase 1)
1. **Preprocessor: FIXED.** Pakai `models/preprocessor.joblib` yang sudah ada, **tidak di-refit**
   saat retraining. Alasan: stabilitas validation gate + konsistensi schema FastAPI.
2. **Signature `train_model()`:** preprocessor di-pass sebagai argument (bukan di-load di dalam).
   ```python
   def train_model(train_df: pd.DataFrame, preprocessor) -> tuple[LogisticRegression, dict]:
       return model, training_metrics
   ```
3. **Threshold: KUNCI di 0.60.** Tidak di-re-tune saat retraining → validation gate apple-to-apple.
   Threshold tuning out of scope sekarang.
4. **Tanggung jawab terpisah (anti data-leakage by design):**
   - `train_model()` HANYA melatih, tidak menyentuh holdout.
   - `evaluate_model()` HANYA evaluasi di holdout.

### Infra decisions
- **Synthetic data:** SDV (`GaussianCopulaSynthesizer`), pre-generated, 5 batch × 100 rows.
  Fit HANYA di `ml/data/train.csv` (tidak pernah di holdout).
- **Model storage:** HuggingFace Hub — repo `MonyetttRindam/telco-churn-models`.
- **Trigger retraining:** FastAPI endpoint + Streamlit button.
- **Validation gate:** model baru di-*promote* hanya jika
  **`F1_new >= F1_current - 0.02` DAN `recall_new >= recall_current`**.
- **Rollback:** manual via endpoint + Streamlit button.
- **Data accumulation:** tiap retraining = data asli + SEMUA batch synthetic sebelumnya.

## Phases

### Phase 1 — Foundation (in progress)
- [Done] 1.1 Update `MLOPS_PLAN.md` (decisions + phases + structure + constraints)
- [Done] 1.2 Setup folder: `ml/`, `ml/data/`, `ml/data/synthetic/`
- [Done] 1.3 `ml/split_holdout.py` → fixed `train.csv` + `test_holdout.csv` (80/20, stratify, rs=42)
- [Done] 1.4 Refactor training: `ml/train.py` (`train_model`) + `ml/evaluate.py` (`evaluate_model`)
- [Done] 1.5 Verifikasi parity: metrics refactor == `models/model_config.json`
- [Done] 1.6 `ml/generate_synthetic.py` → 5 batch SDV + validation report (re-derive engineered features; consistency 0/500)
- [Done] 1.7 `scripts/upload_initial_model.py` → push v_initial + `registry.json` ke HF Hub

### Phase 2 — Backend (2-3 hari)
- [ ] `api/core/model_manager.py` — load/swap model dari HF, rollback
- [ ] `api/core/retraining.py` — pipeline lengkap + validation gate
- [ ] `api/core/registry.py` — baca/tulis `registry.json`
- [ ] Endpoint `/retrain`, `/rollback`, `/status`, `/history`
- [ ] API key auth (simpan di Railway env var)
- [ ] Test semua endpoint via curl/Postman

### Phase 3 — Frontend (1 hari)
- [ ] `streamlit_app/pages/3_MLOps_Dashboard.py`
- [ ] Components: status card, history table, trigger button, rollback button, metric trend chart
- [ ] Deploy ulang ke Railway

### Phase 4 — Polish (opsional)
- [ ] Logging ke file (audit trail)
- [ ] CV bullet + LinkedIn post update

## File Structure (target)

```
telco-churn-prediction/
├── api/                            # (Phase 2) refactor dari app/
│   ├── main.py
│   ├── routes/
│   │   ├── predict.py              # /predict (existing)
│   │   ├── retrain.py              # /retrain, /rollback, /status, /history (new)
│   │   └── auth.py                 # API key middleware
│   └── core/
│       ├── model_manager.py        # load/swap model, HF integration
│       ├── retraining.py           # retrain pipeline + validation gate
│       └── registry.py             # baca/tulis registry.json
│
├── streamlit_app/                  # (Phase 3) refactor dari streamlit_app.py
│   ├── app.py
│   ├── pages/
│   │   ├── 1_Prediction.py
│   │   ├── 2_EDA.py
│   │   └── 3_MLOps_Dashboard.py    # NEW
│   └── utils/
│       └── api_client.py
│
├── ml/                             # (Phase 1) NEW
│   ├── split_holdout.py            # generate fixed train/holdout
│   ├── train.py                    # train_model(train_df, preprocessor)
│   ├── evaluate.py                 # evaluate_model(model, preprocessor, holdout_df)
│   ├── generate_synthetic.py       # SDV script (run sekali)
│   └── data/
│       ├── train.csv               # data asli buat training
│       ├── test_holdout.csv        # FIXED, jangan disentuh
│       └── synthetic/
│           ├── batch_1.csv         # 100 rows
│           ├── batch_2.csv
│           ├── batch_3.csv
│           ├── batch_4.csv
│           └── batch_5.csv
│
├── scripts/
│   └── upload_initial_model.py     # push model v1 + registry.json ke HF
│
├── models/                         # current production (KEEP as reference)
│   ├── model_final.joblib
│   ├── preprocessor.joblib         # FIXED preprocessor (dipakai retraining)
│   ├── split_data.joblib           # LEGACY — superseded by ml/data/*.csv (jangan dihapus)
│   └── model_config.json
│
└── requirements.txt                # +huggingface_hub; requirements-dev.txt +sdv
```

## Critical Constraints

- **Fixed holdout:** `ml/data/test_holdout.csv` di-generate SEKALI dan tidak pernah disentuh lagi.
  Train tidak boleh pernah melihatnya. Semua evaluasi/gate pakai holdout ini.
- **Fixed preprocessor:** `models/preprocessor.joblib` tidak pernah di-refit. Semua model (lama & baru)
  pakai preprocessor yang sama → schema fitur konsisten dengan FastAPI.
- **Fixed threshold:** 0.60, tidak pernah di-re-tune saat retraining.
- **Data accumulation:** retraining ke-N = `train.csv` + `batch_1..N`. Holdout tetap terpisah.
- **Synthetic fit scope:** synthesizer hanya fit di `train.csv`, tidak pernah di holdout.
- **Jangan modify (Phase 1):** `app/`, `streamlit_app.py`, `src/preprocessing.py`.
- **Jangan delete:** `notebooks/`, `models/`, `split_data.joblib` (semua = reference).
- **Legacy note:** `models/split_data.joblib` adalah artifact lama (split in-memory dari notebook 02).
  Digantikan oleh `ml/data/train.csv` + `ml/data/test_holdout.csv`. Disimpan sebagai reference saja.
```

## Decision Log

| Date | Decision | Rationale | Supersedes |
|------|----------|-----------|------------|
| 2026-06-26 | Threshold dikunci di 0.60 | Validation gate jadi apple-to-apple | - |
| 2026-06-26 | Preprocessor fixed (tidak refit) | Stabilitas dimensi fitur + simplicity | - |
| 2026-06-26 | `train_model()` returns metadata only (no performance metrics). Quality assessment is `evaluate_model()`'s responsibility | Separation of Concerns | - |
| 2026-06-26 | `customerID` TIDAK di-drop manual di train/evaluate; preprocessor membuangnya via remainder='drop' | preprocessor.feature_names_in_ memuat customerID (fit di notebook 02); drop manual bikin transform gagal | - |
| 2026-06-26 | SDV `GaussianCopulaSynthesizer`, fit di `train.csv` minus customerID; metadata auto-detect + override `SeniorCitizen`/`Churn`/`is_new_customer`/`has_internet`/`num_addons` jadi categorical | Kolom 0/1 & small-int discrete secara semantic categorical, bukan numeric | - |
| 2026-06-26 | Reproducibility synthetic: `np.random.seed(42)` SEKALI sebelum fit, 5 batch di-sample sekuensial TANPA `reset_sampling()` | RNG maju tiap sample → batch unik (no dup) tapi deterministik antar run (terverifikasi) | - |
| 2026-06-26 | customerID synthetic = `SYNTH_B{batch}_{idx:03d}` | ID asli tak meaningful untuk synthesizer; re-add setelah sampling untuk schema-match | - |
| 2026-06-26 | SDV fit HANYA di base columns (drop 4 engineered); engineered di-derive deterministically via `src.preprocessing.add_features()` setelah sampling | Copula tak bisa menjaga invariant deterministik base↔derived (mis. tenure_group fungsi dari tenure); re-derive memirror pipeline production & respect single-source-of-truth | Override engineered cols jadi categorical (Step 1.6 awal) |
| 2026-06-26 | HF Hub repo `MonyetttRindam/telco-churn-models` (public) sebagai model storage. Token di `.env` (gitignored). Schema: `current/` (active artifacts), `synthetic/` (5 batches), `registry.json` (version history) | Storage model terpusat + versioning untuk retraining; public agar mudah di-pull saat deploy | - |