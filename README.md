
███████╗██████╗  █████╗ ██╗   ██╗██████╗     ██████╗ ███████╗████████╗███████╗ ██████╗████████╗
██╔════╝██╔══██╗██╔══██╗██║   ██║██╔══██╗    ██╔══██╗██╔════╝╚══██╔══╝██╔════╝██╔════╝╚══██╔══╝
█████╗  ██████╔╝███████║██║   ██║██║  ██║    ██║  ██║█████╗     ██║   █████╗  ██║        ██║   
██╔══╝  ██╔══██╗██╔══██║██║   ██║██║  ██║    ██║  ██║██╔══╝     ██║   ██╔══╝  ██║        ██║   
██║     ██║  ██║██║  ██║╚██████╔╝██████╔╝    ██████╔╝███████╗   ██║   ███████╗╚██████╗   ██║   
╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚═════╝     ╚═════╝ ╚══════╝   ╚═╝   ╚══════╝ ╚═════╝   ╚═╝

# Fraud Detection Pipeline

A hybrid ML system that catches fraudulent transactions. Built on 590k real transactions from the IEEE-CIS dataset with Isolation Forest for anomaly detection, LightGBM for classification, SHAP for explainability, all wrapped in a FastAPI backend and a Streamlit dashboard you can actually use.


## Why I built this

Most fraud detection projects on GitHub are a Jupyter notebook with a confusion matrix screenshot and a "future work" section that never happens. I wanted to go further and build the whole thing end-to-end, from raw data to an API endpoint that takes a transaction and returns a verdict with reasons.

The harder question was: what does "good" actually mean for a fraud model? Accuracy is meaningless when 96.5% of transactions are legitimate. A model that just says "not fraud" every time gets 96.5% accuracy and catches nothing. So the entire project is built around a more honest question: *if a fraud analyst gets a queue of flagged transactions, how much of that queue is real fraud, and how much is noise?*


## How it works

Three layers, each doing something different:

**Isolation Forest (unsupervised)** : scores how statistically weird a transaction looks, without ever seeing the fraud labels. This catches patterns the supervised model might miss because it's not constrained to learn from historical fraud. On its own it gets 0.71 AUROC and 7x lift in the top 5% of flagged transactions. Not enough to deploy alone, but useful as a feature.

**LightGBM (supervised)** : the main model. A gradient-boosted classifier trained on 230 engineered features, including the anomaly score from the Isolation Forest. Tuned over 50 Optuna trials. Outputs a fraud probability for every transaction. 0.87 AUROC on the held-out test set.

**SHAP explainability** : every single prediction gets a breakdown of *why*. Not "the model said fraud" but "flagged because: address mismatch, first-time card, unusual amount for this email domain." I didn't add this as a nice-to-have. In financial services, a model that can't explain itself doesn't get deployed the regulators want actual explanations, and analysts needs to know what to investigate.


## The threshold decision

This is probably the most important design choice in the project.

The decision threshold is set at 0.84. That means the model only flags a transaction as fraud when it's really confident. At this threshold:

- 56% of flagged transactions are actually fraud (precision)
- The false alarm rate on legit customers is just 1.1%
- Only 2.4% of all transactions get sent for review

The trade-off? The model misses about 60% of actual fraud cases (recall of 0.40). That sounds bad until you think about it from the operations side. A fraud analyst's time is expensive. A review queue full of false alarms gets ignored. And every legitimate customer who gets declined might leave. I'd rather have a smaller, trustworthy queue where analysts take every alert seriously than a massive queue they learn to tune out.

If you wanted to catch more fraud, you'd lower the threshold, but that means more false alarms, more blocked customers, more analyst fatigue. There's no free lunch here, only a choice about which cost you'd rather pay. I chose precision.


## Results

I evaluated this pipeline on a **time-based test split** the most recent 20% of transactions by date. Not a random split. This simulates what actually happens when you deploy a model: it sees future transactions it wasn't trained on.

| Metric | Score |
|--------|-------|
| ROC-AUC | 0.8733 |
| PR-AUC | 0.4612 (13x better than random) |
| Precision | 0.5640 |
| Recall | 0.3957 |
| F1 | 0.4651 |

Out of 118,108 test transactions: 2,851 flagged for review, 1,608 of 4,064 actual fraud caught, 1,243 legit transactions incorrectly flagged.

The validation-to-test F1 drop (0.56 → 0.47) is real (Honest). That's temporal drift, fraud patterns evolve over time, and a model trained on January data will be slightly worse at catching March fraud. In production, this means you'd need scheduled retraining. That's not a failure, it's how fraud detection actually works.


## What the model looks at

230 engineered features, grouped by what they actually tell you about a transaction:

**Identity match signals** : does the name match the card? Does the billing address check out? M1–M9 are match flags that catch the kind of mismatches you get when someone uses stolen card details.

**Transaction behaviour** : amount, log-transformed amount, amount bins. One thing the model learned that I didn't expect: mid-range amounts are more suspicious than very large ones. Turns out big purchases tend to be legitimate, it's the medium-sized transactions that are more likely to be fraud.

**Usage history and timing** : card usage counts (C1–C14), days since last transaction (D1–D15), hour of day, day of week. A first-time card used at 3am on a Saturday looks very different from a regular customer on a Tuesday afternoon.

**Anomaly score** : how unusual this transaction is compared to the full dataset, from the Isolation Forest.

**Email domain risk** : the historical fraud rate of the sender's email domain. Some domains have meaningfully higher fraud rates.

No single feature decides the outcome. LightGBM combines all 230 across 143 trees, each checking different feature combinations. The final probability is their collective vote.


## Project structure

```
fraud-detection/
├── src/fraud_detection/
│   ├── data/load.py                  # Data loading, merging, validation
│   ├── features/build_features.py    # Feature engineering (5 transformers)
│   ├── models/train.py               # Isolation Forest + LightGBM + Optuna
│   ├── evaluation/metrics.py         # Metrics, threshold optimization, SHAP
│   ├── api/app.py                    # FastAPI inference endpoint
│   └── dashboard/app.py              # Streamlit interactive dashboard
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

```bash
git clone https://github.com/vshal999/fraud-detection.git
cd fraud-detection
python -m venv venv
```

Activate:

```bash
# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

Install:

```bash
pip install -r requirements.txt
```

For this project I have made use of the dataset from [Kaggle IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection/data). Grab these two files and put them in `data/raw/`:

- `train_transaction.csv`
- `train_identity.csv`

The Kaggle test files don't have labels and they're not used here.


## Running the pipeline

Want to get raw results on a different dataset? Train everything from scratch:

```bash
python scripts/run_pipeline.py
```

This basically re-runs: data loading → feature engineering → Isolation Forest → LightGBM with Optuna → evaluation → saves models and plots. Takes 30–45 minutes depending on your machine.

Run the tests:

```bash
pytest tests/ -v
```

112 should pass.


## Using the dashboard

The easiest way to interact with the model. You need two terminals.

**Terminal 1 — API backend:**

```bash
uvicorn fraud_detection.api.app:app --reload --port 8000
```

Wait for `Application startup complete`.

**Terminal 2 — Streamlit dashboard:**

```bash
streamlit run src/fraud_detection/dashboard/app.py
```

Opens at `http://localhost:8501`. Fill in transaction details (anything blank gets filled with training medians), hit **Analyse Transaction**, and you get:

- Fraud probability (0–1)
- Verdict — red for fraud (above 0.84), green for legit
- Top 3 risk factors explaining the decision
- SHAP waterfall chart showing each feature's contribution

`Ctrl + C` in both terminals to stop.


## Using the API directly

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

Interactive API docs at `http://localhost:8000/docs`.


## Tech stack

Python 3.10+, pandas, numpy, scikit-learn, LightGBM, Optuna, SHAP, imbalanced-learn, FastAPI, Streamlit, pytest, ruff


## License

MIT
