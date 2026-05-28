# CLAUDE.md — Fraud Detection Pipeline

## Project Overview
End-to-end fraud detection pipeline using the IEEE-CIS Fraud Detection dataset (590k transactions). Portfolio-grade: production-ready code, tested, documented, deployable.

## Dataset
- Source: Kaggle IEEE-CIS Fraud Detection
- Two tables: `transaction` (transactions with features) and `identity` (identity info linked by TransactionID)
- Target: `isFraud` (binary, ~3.5% positive rate — heavily imbalanced)
- Features: V1-V339 (anonymous), C1-C14 (counting), D1-D15 (timedelta), M1-M9 (match), card1-card6, addr1-addr2, email domain, device info

## Architecture
```
src/fraud_detection/
├── data/           → loading, splitting, validation
├── features/       → feature engineering, transformers
├── models/         → training, hyperparameter tuning, model registry
├── evaluation/     → metrics, threshold optimization, reporting
└── api/            → FastAPI inference endpoint
```

## Key Technical Decisions
- All preprocessing inside sklearn Pipelines — no leakage
- Stratified splits everywhere (3.5% fraud rate)
- Walk-forward time-based split (TransactionDT) for final evaluation
- Hybrid approach: Isolation Forest (unsupervised) + LightGBM (supervised) + rule-based flags
- Threshold optimization on precision-recall curve, not default 0.5
- SHAP for model interpretability

## Build Commands
```bash
pip install -r requirements.txt          # install deps
pytest tests/ -v                         # run tests
python -m fraud_detection.data.load      # download and prepare data
python -m fraud_detection.models.train   # train pipeline
python scripts/run_pipeline.py           # full end-to-end
uvicorn fraud_detection.api.app:app      # serve inference API
```

## Conventions
- Type hints on all functions
- Docstrings on all public functions
- Logging over print statements
- Config in config/config.yaml — no magic numbers in code
- Conventional commits: feat:, fix:, docs:, refactor:, test:
