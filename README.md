# Fraud Detection Pipeline

A hybrid machine learning system that detects fraudulent financial transactions. Built with the IEEE-CIS Fraud Detection dataset (590k real transactions), this project combines unsupervised anomaly detection, supervised classification, and model explainability into a deployable product with a live dashboard.

This isn't a notebook exercise. It's a full pipeline — from raw data to a working API and interactive frontend where you can punch in a transaction and get a fraud verdict with reasons in real time.


## How It Works

The system runs three layers of detection on every transaction:

**Layer 1 — Isolation Forest (unsupervised).** Scores how "weird" a transaction looks compared to everything else, without knowing which ones are actually fraud. Catches novel patterns that supervised models miss. On its own it achieves 0.71 AUROC and 7x lift at the top 5% of flagged transactions.

**Layer 2 — LightGBM (supervised).** A gradient-boosted classifier trained on labelled fraud/legit examples. Tuned with 50 Optuna trials. Takes all 230 engineered features (including the anomaly score from Layer 1) and outputs a fraud probability. This is the main workhorse — 0.87 AUROC on the held-out test set.

**Layer 3 — SHAP explainability.** Every prediction comes with a breakdown of which features pushed the score up or down. Instead of "the model said fraud," you get "flagged because: address mismatch, first-time card use, unusual transaction amount."

The decision threshold is set at 0.84 — the model only flags a transaction as fraud when it's confident. At this threshold, 56% of flagged transactions are real fraud, and the false alarm rate on legit transactions is just 1.1%.


## Results

Evaluated on a time-based test split (the most recent 20% of transactions by date, simulating how it would perform on future data):

| Metric | Score |
|--------|-------|
| ROC-AUC | 0.8733 |
| PR-AUC | 0.4612 (13x better than random) |
| Precision | 0.5640 |
| Recall | 0.3957 |
| F1 | 0.4651 |

Out of 118,108 test transactions, the model flagged 2,851 for review (2.4% of all transactions), caught 1,608 of 4,064 actual fraud cases, and incorrectly flagged 1,243 legit transactions. The trade-off is intentional — high precision means the review queue is mostly real fraud, not noise.

The val-to-test performance drop (F1 0.56 → 0.47) reflects temporal drift — fraud patterns evolve over time, which is expected and realistic.


## What the Model Looks At

The pipeline engineers 230 features from the raw data, grouped by what they reveal about a transaction:

**Identity signals** — does the name match the card? Does the billing address check out? M1 through M9 match flags catch mismatches that fraudsters typically trigger when using stolen card details.

**Transaction behavior** — amount, amount bin, log-transformed amount. The model learned that mid-range amounts are actually more suspicious than extreme ones (very high amounts tend to be legit big purchases).

**History and timing** — how many times this card has been used (C1-C14), days since last transaction (D1-D15), hour of day, day of week, weekend flag. A first-time card used at 3am on a weekend looks different from a regular customer on a Tuesday afternoon.

**Anomaly score** — how statistically unusual this transaction is compared to the rest of the dataset, as scored by the Isolation Forest.

**Email domain risk** — the historical fraud rate of the payer's email domain. Some domains have significantly higher fraud rates than others.

No single feature determines the outcome. LightGBM combines all 230 features across 143 decision trees, each checking different combinations, and the final probability is their collective vote.


## Project Structure

```
fraud-detection/
├── src/fraud_detection/
│   ├── data/load.py                  # Data loading, merging, validation
│   ├── features/build_features.py    # Feature engineering pipeline (5 transformers)
│   ├── models/train.py               # Isolation Forest + LightGBM + Optuna tuning
│   ├── evaluation/metrics.py         # Metrics, threshold optimization, SHAP
│   ├── api/app.py                    # FastAPI inference endpoint
│   └── dashboard/app.py             # Streamlit interactive dashboard
├── config/config.yaml                # All model params, paths, thresholds
├── data/
│   ├── raw/                          # Original Kaggle CSVs (not committed)
│   ├── interim/                      # Intermediate files
│   └── processed/                    # Model-ready parquet files
├── models/                           # Saved model artifacts
├── notebooks/                        # Exploration and analysis notebooks
├── reports/figures/                   # Evaluation plots
├── tests/                            # 112 tests (unit + integration)
└── scripts/run_pipeline.py           # End-to-end execution
```


## Setup

Clone the repo and install dependencies:

```bash
git clone https://github.com/wn25351/fraud-detection.git
cd fraud-detection
python -m venv venv
```

Activate the virtual environment:

```bash
# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

Install everything:

```bash
pip install -r requirements.txt
```

Download the dataset from [Kaggle IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection/data). You only need two files — place them in `data/raw/`:

- `train_transaction.csv`
- `train_identity.csv`

(The test files from Kaggle don't have labels and aren't used.)


## Running the Pipeline

Train everything from scratch:

```bash
python scripts/run_pipeline.py
```

This runs data loading → feature engineering → Isolation Forest → LightGBM training with Optuna → evaluation → saves all models and plots. Takes roughly 30-45 minutes depending on your machine.

Run the tests to make sure everything is working:

```bash
pytest tests/ -v
```

Should see 112 passed.


## Using the Dashboard

The dashboard is the easiest way to interact with the model. You need two terminals running.

**Terminal 1 — start the API backend:**

```bash
uvicorn fraud_detection.api.app:app --reload --port 8000
```

Wait until you see `Application startup complete`. The `--reload` flag means it auto-restarts if you edit the code.

**Terminal 2 — start the Streamlit dashboard:**

```bash
streamlit run src/fraud_detection/dashboard/app.py
```

It'll open in your browser at `http://localhost:8501`. If it doesn't open automatically, click the URL in the terminal.

On the dashboard you'll see input fields for transaction details — amount, card network, card type, email domain, match flags, and more. Fill in the details (anything you leave blank gets filled with training data medians), hit **Analyse Transaction**, and you'll get:

- A fraud probability score (0 to 1)
- A verdict — red for fraud (above 0.84 threshold), green for legit
- Top 3 risk factors explaining why
- A SHAP waterfall chart showing how each feature contributed

To stop everything, press `Ctrl + C` in both terminals.


## Using the API Directly

If you want to call the API programmatically (from another service, script, or tool):

```bash
uvicorn fraud_detection.api.app:app --reload --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
```

Predict:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"TransactionAmt": 150, "card4": "visa", "card6": "credit", "ProductCD": "W", "P_emaildomain": "gmail.com", "M1": "T", "M4": "M0", "C1": 1, "D1": 30, "card1": 10000}'
```

You can also open `http://localhost:8000/docs` in your browser for an interactive API explorer where you can test requests without curl.


## Tech Stack

Python 3.10+, pandas, numpy, scikit-learn, LightGBM, Optuna, SHAP, imbalanced-learn, FastAPI, Streamlit, pytest, ruff


## License

MIT
