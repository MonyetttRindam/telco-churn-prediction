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

Keputusan Phase 2 (locked):
- **Batch selection:** Upload + manual dropdown.
- **Task handling:** Background task (retraining async).
- **Archive strategy:** Registry reference (1 registry file, versi ditrack di dalamnya).
- **Rejection handling:** Model yang gagal gate diarsipkan status `rejected`.
- **Validation rules:** Ketat (7 rules pada batch upload).
- **Batch naming:** Auto-rename `batch_N.csv` (via `next_batch_number`).
- **Concurrency:** Thread lock (single-instance).

- [Done] 2.1 `api/core/registry.py` — kelas `Registry` (satu-satunya interface `registry.json`), thread-safe + cache TTL; schema v2 (versions + batches); `scripts/migrate_registry_v2.py` (one-time, run + verified); `api/core/test_registry.py` (manual test, self-cleanup, all PASS)
- [Done] 2.2 `api/core/model_manager.py` — load/swap model dari HF (atomic), thread-safe (RLock); HF path convention `models/{version_id}/`; `scripts/migrate_hf_structure_v2.py` (one-time, run + verified); `api/core/test_model_manager.py` (all PASS, live)
- [Done] 2.3 `api/core/batch_manager.py` — kelas `BatchManager` (single source of truth batch di HF); list/download/validate(7 rules)/upload; vocabulary (16 categorical) di-extract dari `train.csv` di `__init__`; `api/core/test_batch_manager.py` (9/9 PASS, live)
- [Done] 2.4 `api/core/retraining.py` — `RetrainingPipeline.retrain(batch_id)`: accumulate → reuse `ml/train.py`+`ml/evaluate.py` → validation gate → promote/reject → upload `models/{version_id}/`; `api/core/test_retraining.py` (live: batch_1 REJECTED by recall gate, cleanup 0 stray)
- [Done] 2.5 `api/core/job_queue.py` — in-memory background job queue (daemon thread, `Lock`, LRU eviction oldest-terminal, dataclass→asdict result serialization); `api/core/test_job_queue.py` (7/7 PASS, pure local)
- [Done] 2.6 `api/core/auth.py` — API key auth (`X-API-Key` header, `MLOPS_API_KEY` di `.env`/Railway, constant-time compare, 401/403/500); `.env.example` template; `api/core/test_auth.py` (4/4 PASS, TestClient)
- [Done] 2.7 Router layer: `api/deps.py` (singleton DI), `api/schemas.py` (Pydantic v2), `api/routes/{mlops,retrain,batches,jobs}.py` (`/status`,`/history`,`/rollback`,`/verify-key`,`/retrain`,`/batches`,`/upload-batch`,`/jobs`); exception→HTTP mapping; `api/test_endpoints.py` (11/11 PASS live, self-cleanup)
- [Done] 2.8 Integrasi `app/main.py`: lifespan startup (`init_dependencies`), `/predict` pakai singleton `ModelManager` (active version), include 4 router Phase 2, tambah `model_version` di `PredictionOutput`; README API Usage section; verified live (startup clean, 4 curl PASS, `/predict` → model_version=v_initial)

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
| 2026-07-02 | `registry.json` schema v2: tambah field `batches.available[]` + `batches.next_batch_number`; track versions DAN batches dalam satu file (registry-reference archive strategy) | Retraining butuh tahu batch mana available/unused & auto-naming `batch_N`; satu file = simple, atomic | schema v1 (versions only) |
| 2026-07-02 | `Registry` = satu-satunya interface `registry.json`; thread-safe (`Lock`), in-memory cache TTL 60s, WRITE selalu re-read fresh dari Hub sebelum mutate lalu invalidate cache | Konsistensi single-instance + hindari clobber stale write; cache kurangi latency read berulang | - |
| 2026-07-02 | Version status lifecycle: `pending`/`rejected` → `active` (promote), old active → `archived`; `previous` = target rollback. Hanya `pending`/`rejected` yang promotable | Gate menghasilkan model pending; reject diarsipkan; rollback butuh pointer `previous` | - |
| 2026-07-02 | **Known optimization (LOW PRIORITY, untuk Step 2.7 UI responsiveness):** cache `BatchManager._duplicate_reference()` dengan TTL ~5 menit. Saat ini reference (train + semua batch) di-download & dibangun ulang tiap `validate_batch()` call — lambat kalau UI validasi berkali-kali. Belum diimplement; catat sebagai perbaikan | Dedup reference jarang berubah dalam window pendek; TTL cache kurangi latensi validasi berulang di endpoint upload | - |
| 2026-07-02 | Retraining: model di-upload ke `models/{version_id}/`; preprocessor FIXED tetap di-upload ulang tiap versi (redundan) agar tiap versi self-contained & loadable independen oleh `ModelManager` | Konsistensi load per-versi > hemat storage (preprocessor 8KB); hindari special-case saat swap/rollback | - |
| 2026-07-02 | Retraining order: upload artifacts → `add_version(pending)` → `promote_version` → `model_manager.swap_to` → `mark_batch_used`. Prefer konsistensi registry > in-memory (kalau swap gagal, registry sudah promoted, `/predict` masih pakai model lama sampai reload) | Registry = source of truth; in-memory recoverable via `load_active()` | - |