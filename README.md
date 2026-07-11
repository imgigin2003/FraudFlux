# 💳 FraudFlux

> **A false alarm costs a phone call. A missed fraud costs someone's account.**

FraudFlux is a fraud detection pipeline built on 284,807 real European credit card transactions, designed around the same non-negotiable rule as my other project, AffectCare: **missing a real emergency is worse than a false alarm — until it isn't.** This project is also the story of where that rule actually breaks down, and what I did instead.

---

## ✨ What it does

Give it a CSV of new transactions. It tells you which ones look like fraud.

```bash
python predict.py path/to/transactions.csv
```

```
Transaction 12: 96.7% fraud probability -> FRAUD
Transaction 13: 0.0% fraud probability -> Not Fraud
```

---

## 📉 The problem with "recall over precision, always"

I started this project the same way I approached AffectCare: catching every fraud matters more than avoiding false alarms, full stop. That instinct is correct in principle — a missed fraud (false negative) can drain someone's account; a false alarm (false positive) is just an annoyance.

But principles need to survive contact with real numbers. Here's what happened when I actually followed "recall over precision" to its logical end:

| Model                          | Threshold | Recall    | Precision | False Alarms |
| ------------------------------ | --------- | --------- | --------- | ------------ |
| Logistic Regression (balanced) | 0.7       | **87.4%** | 11.7%     | **629**      |
| Random Forest                  | 0.15      | 81.1%     | 80.2%     | **19**       |

Logistic Regression catches 5 more real frauds out of 95. It also generates **33x more false alarms** doing it. In a real deployed system, that's not "extra caution" — that's a fraud alert that gets ignored within a week, because it cries wolf 629 times to catch 83 real cases. A doctor's alarm nobody trusts anymore isn't a safety net; it's noise.

**I ended up picking the Random Forest at threshold 0.15** — balanced ~80% recall and ~80% precision, only 19 false alarms out of 56,746 test transactions. Not the model with the single highest recall number. The one that would actually survive being deployed.

This is the real lesson of the project, and it directly contradicts how I described my own approach on AffectCare's README. I'm leaving that contradiction visible rather than quietly editing my past self — a stated principle is only as good as what happens when you check its cost in real numbers.

---

## 📊 Results

Final model: **Random Forest (`class_weight="balanced"`, `n_estimators=150`), threshold = 0.15**, evaluated on a stratified 20% held-out test split (56,746 transactions, 95 real frauds):

| Metric                  | Score      |
| ----------------------- | ---------- |
| Recall (frauds caught)  | **81.05%** |
| Precision (real alerts) | 80.21%     |
| F1 Score                | 80.63%     |

**Confusion Matrix**

```
[[56632    19]      TN=56632   FP=19
 [   18    77]]      FN=18      TP=77
```

77 real frauds caught, 18 missed, 19 false alarms out of 56,651 legitimate transactions. Not perfect — no fraud model is — but a genuinely deployable tradeoff, arrived at by comparing four real candidates rather than assuming the fanciest model wins.

### Every model I actually tried, and why each lost

| Model                                           | Best Threshold | Recall    | Precision | Verdict                                                                                              |
| ----------------------------------------------- | -------------- | --------- | --------- | ---------------------------------------------------------------------------------------------------- |
| Logistic Regression (`class_weight="balanced"`) | 0.7            | 87.4%     | 11.7%     | Highest recall, but 629 false alarms — undeployable                                                  |
| Decision Tree (`max_depth=5`, hand-picked)      | 0.9            | 81.1%     | 15.2%     | Better than LR on precision, still noisy                                                             |
| Decision Tree (GridSearchCV, F2-tuned)          | any 0.3–0.9    | 77.9%     | 23.2%     | Leaf-based probabilities cluster at ~12 discrete values — threshold sweeping barely moves the needle |
| **Random Forest**                               | **0.15**       | **81.1%** | **80.2%** | **Winner** — smoothest probability distribution (70+ unique values), best balance                    |

---

## 🔍 Dataset

284,807 European credit card transactions from September 2013, over two days. **492 are fraud — 0.172% of the total.** This single number drives almost every decision in the pipeline:

- A model that predicts "not fraud" for every transaction scores **99.8% accuracy** while catching zero fraud — accuracy is actively misleading here, which is why every evaluation in this project reports precision/recall/F1 instead
- `stratify=y` in the train/test split, so the 0.172% fraud ratio is preserved in both sets — a random split risks a test set with almost no fraud examples at all, purely by chance
- `class_weight="balanced"` on every model, forcing the training loss to penalize a missed fraud far more heavily than a missed non-fraud, proportional to how rare fraud actually is
- 1,081 exact duplicate rows found and dropped **before** the train/test split — leaving them in risked the same fraud transaction appearing in both train and test, silently inflating the test score

`V1`–`V28` are already PCA-transformed by the original dataset publishers (anonymized banking features) and arrive pre-scaled. Only `Time` and `Amount` needed `StandardScaler` — scaling the PCA columns again would have been redundant, not wrong, but a sign of not understanding why scaling exists in the first place.

---

## 🧪 Key design decisions

**Threshold tuning happened _after_ model selection, for each model separately, because tree-based models don't produce smooth probabilities.** Logistic Regression's `predict_proba()` output is a continuous sigmoid — sweeping thresholds from 0.3 to 0.9 gives meaningfully different results at every step. A single unconstrained Decision Tree's leaves are pure or near-pure, so `predict_proba()` clusters at a handful of fixed values (mostly exact 0.0 and 1.0) — sweeping thresholds across that range did almost nothing, which looked like a bug the first time I saw it and turned out to be a real, documented property of how trees estimate probability.

**GridSearchCV scored on `fbeta_score(beta=2)`, not plain recall.** Scoring purely on recall handed the search an unconstrained tree (`max_depth=None, min_samples_leaf=1`) — the most overfit option available, since recall alone doesn't penalize the flood of false positives that configuration produces. Switching to F2 (recall weighted 2x over precision, closer to this project's actual priorities than plain F1) forced the search toward genuinely constrained, more honest trees.

**`n_jobs=-1` on every GridSearchCV and RandomForestClassifier call.** The first GridSearchCV run took 10 minutes 49 seconds on a single core. Adding `n_jobs=-1` — parallelizing across all available CPU cores — cut it to 2 minutes with identical results. Not a modeling decision, but a real, measurable engineering one.

---

## ⚠️ Known limitation (documented, not fixed)

`predict.py` currently loads the trained model from a `.npy` array (no column names attached), but feeds it a pandas DataFrame at inference time (which does carry column names). Scikit-learn correctly warns about this mismatch — it can't verify the incoming columns are in the same order the model was trained on, since it was never given names to check against in the first place.

In this project's case, the column order is manually kept consistent between `prepare_data.py` and `predict.py`, so predictions are currently correct — verified against known fraud/non-fraud patterns from the source dataset. But this is fragile: a future column reorder, or a different person calling `predict.py` with a differently-ordered CSV, would fail silently with no error, just an easy-to-ignore warning. The proper fix is training directly on a labeled DataFrame instead of a converted `.npy` array, so scikit-learn can validate column names at inference time. Not fixed here due to time constraints ahead of a grad school application deadline — named honestly instead of silently suppressed.

---

## 📁 Project structure

```
FraudFlux/
├── dataset/
│   └── testcreditcard.csv          # test data
├── data/
│   └── processed/              # X_train.npy, X_test.npy, y_train.npy, y_test.npy, scaler.pkl
├── src/
│   ├── prepare_data.py         # load → dedupe → split (stratified) → scale (Time, Amount only) → save
│   ├── train.py                # Logistic Regression, Decision Tree, Random Forest — all with class_weight="balanced"
│   └── evaluate.py             # per-model threshold sweeps, confusion matrices, the comparison table above
├── models/
│   ├── baseline_model.pkl
│   ├── decision_tree_model.pkl
│   └── random_forest.pkl       # ← the deployed model
├── predict.py                  # ← run this. inference on new, unlabeled transactions.
└── README.md
```

---

## ⚙️ Installation & usage

```bash
git clone https://github.com/imgigin2003/FraudFlux.git
cd FraudFlux
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r src/requirements.txt
```

**Run a prediction:**

```bash
python predict.py path/to/transactions.csv
```

**Retrain from scratch:**

```bash
cd src
python3 prepare_data.py    # cleans + splits + scales creditcard.csv, saves to data/processed/
python3 train.py           # trains all three models, saves to models/
python3 evaluate.py        # runs threshold sweeps, prints comparison metrics
```

---

## 🗣️ Talking points (things I can actually explain, unscripted)

- Why 99.8% accuracy is a meaningless number on this dataset, and what I checked instead
- Why I dropped duplicates _before_ splitting, not after — and what would have leaked if I hadn't
- Why a single Decision Tree's threshold sweep did nothing, and what that revealed about how tree-based `predict_proba()` actually works
- Why GridSearchCV scored on recall alone picked the most overfit tree in the entire grid, and why switching to F2 fixed it
- Why I abandoned "recall over precision, always" once I saw what 629 false alarms actually looks like next to 19 — and why that reversal is the most honest part of this README, not something to hide

---

## 🤝 Notes

Built as a solo learning project, deliberately connecting concepts I'd already learned individually (train/test leakage, scaling, threshold tuning, GridSearchCV) into one real end-to-end pipeline for the first time, on a dataset chosen to mirror my next real project. The known limitation above is named on purpose — an honest account of what's unfinished is worth more than a repo that looks cleaner than it is.

---

_Built with scikit-learn, pandas, and a genuine change of mind about what "recall-first" actually means in practice._
