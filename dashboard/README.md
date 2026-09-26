# FraudFlux Dashboard

A live transaction monitor for the trained Random Forest model (threshold 0.15).
Upload a CSV of unlabeled transactions, the API scores every row, and the UI streams
verdicts into the feed with running stats. Click any row to see why the model
reached that verdict (per-transaction SHAP feature contributions).

```
dashboard/
├── backend/          # FastAPI, wraps models/random_forest.pkl + data/processed/scaler.pkl
│   ├── main.py
│   └── requirements.txt
└── frontend/         # React + Vite, styled after FraudFluxPreview
    ├── index.html
    ├── package.json
    ├── vite.config.js
    └── src/
```

## Run the backend

Run from the repo root so the default model and scaler paths resolve.

```bash
pip install -r dashboard/backend/requirements.txt
uvicorn dashboard.backend.main:app --reload --port 8000
```

Optional environment variables:

| Variable | Default |
| --- | --- |
| `FRAUDFLUX_MODEL_PATH` | `./models/random_forest.pkl` |
| `FRAUDFLUX_SCALER_PATH` | `./data/processed/scaler.pkl` |
| `FRAUDFLUX_THRESHOLD` | `0.15` |
| `FRAUDFLUX_CORS_ORIGINS` | `http://localhost:5173` |
| `FRAUDFLUX_MAX_ROWS` | `200000` |
| `FRAUDFLUX_MAX_CACHED_JOBS` | `5` (scored uploads kept in memory for explanations) |

Endpoints:

- `GET /api/health` -> model, scaler, and explainer load status
- `GET /api/model` -> model name, threshold, expected columns, reported test metrics
- `POST /api/predict` (multipart `file`, optional `threshold` form field) -> `job_id`, per-row probability and verdict, summary
- `GET /api/explain/{job_id}/{index}` -> SHAP contributions for one scored row: baseline probability, signed contribution and raw value per feature, plain-language summary

Explanations use `shap.TreeExplainer` on the loaded Random Forest. If `shap` is not
installed the API still scores; the explain endpoint returns 503 and the UI shows
an error with a retry button.

## Run the frontend

```bash
cd dashboard/frontend
npm install
npm run dev
```

Open http://localhost:5173. Vite proxies `/api` to the backend on port 8000.

## CSV format

The CSV must contain exactly the model's 30 feature columns, in any order:
`Time, V1 ... V28, Amount`. Extra columns (including `Class`) are rejected so
scoring never runs on a differently shaped file by accident.
