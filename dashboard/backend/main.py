"""FraudFlux scoring API.

Wraps the deployed Random Forest (models/random_forest.pkl) and the StandardScaler
fitted in src/prepare_data.py. Mirrors predict.py, but returns JSON instead of
printing, and enforces the exact column set so the known column-order limitation
documented in the README cannot fail silently.

Also exposes per-transaction explanations (SHAP TreeExplainer) so the dashboard
can show which features pushed a row toward or away from fraud.
"""

import io
import os
import uuid
from collections import OrderedDict
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

MODEL_PATH = os.getenv("FRAUDFLUX_MODEL_PATH", "./models/random_forest.pkl")
SCALER_PATH = os.getenv("FRAUDFLUX_SCALER_PATH", "./data/processed/scaler.pkl")
DEFAULT_THRESHOLD = float(os.getenv("FRAUDFLUX_THRESHOLD", "0.15"))
CORS_ORIGINS = os.getenv("FRAUDFLUX_CORS_ORIGINS", "http://localhost:5173").split(",")
MAX_ROWS = int(os.getenv("FRAUDFLUX_MAX_ROWS", "200000"))
MAX_CACHED_JOBS = int(os.getenv("FRAUDFLUX_MAX_CACHED_JOBS", "5"))

# Exact training column order from the source dataset (Class dropped).
FEATURE_COLUMNS: List[str] = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
COLUMNS_TO_SCALE = ["Time", "Amount"]

# Metrics reported in the project README for the held-out test split.
REPORTED_METRICS = {
    "model": "Random Forest",
    "recall": 0.8105,
    "precision": 0.8021,
    "f1": 0.8063,
    "test_transactions": 56746,
    "confusion_matrix": {"tn": 56632, "fp": 19, "fn": 18, "tp": 77},
}

app = FastAPI(title="FraudFlux API", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None
_scaler = None
_explainer = None
_load_error: Optional[str] = None
_explainer_error: Optional[str] = None

# Most recent scored uploads, so /api/explain can look a row up by job id.
# Each value: {"raw": DataFrame, "scaled": ndarray, "probs": ndarray, "threshold": float}
_jobs: "OrderedDict[str, dict]" = OrderedDict()


@app.on_event("startup")
def load_artifacts() -> None:
    global _model, _scaler, _explainer, _load_error, _explainer_error
    try:
        _model = joblib.load(MODEL_PATH)
        _scaler = joblib.load(SCALER_PATH)
        _load_error = None
    except Exception as exc:  # noqa: BLE001
        _load_error = f"{type(exc).__name__}: {exc}"
        return

    try:
        import shap  # imported lazily so the API still scores without it

        _explainer = shap.TreeExplainer(_model)
        _explainer_error = None
    except Exception as exc:  # noqa: BLE001
        _explainer = None
        _explainer_error = f"{type(exc).__name__}: {exc}"


def _require_artifacts():
    if _model is None or _scaler is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model or scaler not loaded. {_load_error or ''}".strip(),
        )


def _validate_columns(df: pd.DataFrame) -> pd.DataFrame:
    incoming = list(df.columns)
    missing = [c for c in FEATURE_COLUMNS if c not in incoming]
    extra = [c for c in incoming if c not in FEATURE_COLUMNS]
    if missing or extra:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "CSV columns do not match the model's 30 features.",
                "missing": missing,
                "unexpected": extra,
                "expected": FEATURE_COLUMNS,
            },
        )
    # Reorder to the training order so the numpy matrix matches the model.
    return df[FEATURE_COLUMNS]


def _scale(df: pd.DataFrame) -> np.ndarray:
    scaled = df.copy()
    scaled[COLUMNS_TO_SCALE] = _scaler.transform(df[COLUMNS_TO_SCALE])
    # The model was fitted on a numpy array, so score a numpy array too.
    # This avoids the feature-name warning noted in the README.
    return scaled.to_numpy()


def _remember_job(job: dict) -> str:
    job_id = uuid.uuid4().hex
    _jobs[job_id] = job
    while len(_jobs) > MAX_CACHED_JOBS:
        _jobs.popitem(last=False)
    return job_id


def _fraud_class_index() -> int:
    classes = list(getattr(_model, "classes_", [0, 1]))
    return classes.index(1) if 1 in classes else len(classes) - 1


def _shap_for_row(x_row: np.ndarray):
    """Return (contributions[30], baseline) for the fraud class.

    Handles both shap output layouts: a list of per-class arrays (older shap)
    and a single array shaped (n_samples, n_features, n_classes) (newer shap).
    """
    k = _fraud_class_index()
    values = _explainer.shap_values(x_row.reshape(1, -1))
    expected = _explainer.expected_value

    if isinstance(values, list):
        contrib = np.asarray(values[k])[0]
        base = float(np.asarray(expected)[k]) if np.ndim(expected) else float(expected)
    else:
        arr = np.asarray(values)
        if arr.ndim == 3:
            contrib = arr[0, :, k]
            base = float(np.asarray(expected)[k])
        else:
            contrib = arr[0]
            base = float(np.asarray(expected).ravel()[0])
    return contrib.astype(float), base


@app.get("/api/health")
def health():
    return {
        "status": "ok" if _load_error is None else "degraded",
        "model_loaded": _model is not None,
        "scaler_loaded": _scaler is not None,
        "explainer_loaded": _explainer is not None,
        "error": _load_error,
        "explainer_error": _explainer_error,
    }


@app.get("/api/model")
def model_info():
    info = {
        "threshold": DEFAULT_THRESHOLD,
        "expected_columns": FEATURE_COLUMNS,
        "scaled_columns": COLUMNS_TO_SCALE,
        "explanations_available": _explainer is not None,
        **REPORTED_METRICS,
    }
    if _model is not None:
        info["estimator"] = type(_model).__name__
        info["n_estimators"] = getattr(_model, "n_estimators", None)
    return info


@app.post("/api/predict")
async def predict(
    file: UploadFile = File(...),
    threshold: Optional[float] = Form(None),
):
    _require_artifacts()

    thr = DEFAULT_THRESHOLD if threshold is None else float(threshold)
    if not 0.0 <= thr <= 1.0:
        raise HTTPException(status_code=422, detail="threshold must be between 0 and 1")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=422, detail="Uploaded file is empty.")

    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=f"Could not parse CSV: {exc}")

    if len(df) == 0:
        raise HTTPException(status_code=422, detail="CSV has a header but no rows.")
    if len(df) > MAX_ROWS:
        raise HTTPException(
            status_code=413, detail=f"CSV has {len(df)} rows; limit is {MAX_ROWS}."
        )

    df = _validate_columns(df)

    if df.isnull().any().any():
        bad = df.columns[df.isnull().any()].tolist()
        raise HTTPException(
            status_code=422, detail={"message": "CSV contains empty cells.", "columns": bad}
        )

    try:
        df = df.astype(float).reset_index(drop=True)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"Non-numeric value in CSV: {exc}")

    x_scaled = _scale(df)
    probs = _model.predict_proba(x_scaled)[:, _fraud_class_index()]
    preds = (probs >= thr).astype(int)

    job_id = _remember_job(
        {"raw": df, "scaled": x_scaled, "probs": probs, "threshold": thr}
    )

    rows = [
        {
            "index": int(i),
            "time": float(df["Time"].iat[i]),
            "amount": float(df["Amount"].iat[i]),
            "probability": round(float(probs[i]), 4),
            "is_fraud": bool(preds[i]),
        }
        for i in range(len(df))
    ]

    n_fraud = int(preds.sum())
    return {
        "job_id": job_id,
        "filename": file.filename,
        "threshold": thr,
        "explanations_available": _explainer is not None,
        "summary": {
            "total": len(df),
            "fraud": n_fraud,
            "not_fraud": len(df) - n_fraud,
            "fraud_rate": round(n_fraud / len(df), 6),
            "flagged_amount": round(float(df["Amount"][preds == 1].sum()), 2),
            "mean_probability": round(float(np.mean(probs)), 6),
            "max_probability": round(float(np.max(probs)), 6),
        },
        "rows": rows,
    }


@app.get("/api/explain/{job_id}/{index}")
def explain(job_id: str, index: int):
    _require_artifacts()
    if _explainer is None:
        raise HTTPException(
            status_code=503,
            detail=f"Explanations unavailable. Install `shap`. {_explainer_error or ''}".strip(),
        )

    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail="Scored file no longer cached. Upload it again to explain rows.",
        )
    if not 0 <= index < len(job["raw"]):
        raise HTTPException(status_code=404, detail="Transaction index out of range.")

    try:
        contrib, baseline = _shap_for_row(job["scaled"][index])
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Explanation failed: {exc}")

    raw_row = job["raw"].iloc[index]
    prob = float(job["probs"][index])
    thr = float(job["threshold"])

    features = [
        {
            "feature": name,
            "value": round(float(raw_row[name]), 6),
            "contribution": round(float(contrib[j]), 6),
        }
        for j, name in enumerate(FEATURE_COLUMNS)
    ]
    features.sort(key=lambda f: abs(f["contribution"]), reverse=True)

    top = features[:3]
    is_fraud = prob >= thr
    parts = [
        f"{f['feature']} pushed {'toward' if f['contribution'] > 0 else 'away from'} fraud "
        f"({f['contribution']:+.3f})"
        for f in top
    ]
    if is_fraud:
        summary = (
            f"Flagged: probability {prob:.3f} is at or above the {thr:.2f} threshold. "
            + "; ".join(parts)
            + "."
        )
    else:
        holders = [f for f in features if f["contribution"] < 0][:3]
        held = ", ".join(f["feature"] for f in holders) or "no single feature"
        summary = (
            f"Not flagged: probability {prob:.3f} stayed below the {thr:.2f} threshold. "
            f"Kept low mainly by {held}. "
            + "; ".join(parts)
            + "."
        )

    return {
        "index": index,
        "amount": float(raw_row["Amount"]),
        "time": float(raw_row["Time"]),
        "probability": round(prob, 4),
        "threshold": thr,
        "is_fraud": is_fraud,
        "baseline": round(baseline, 6),
        "features": features,
        "summary": summary,
    }
