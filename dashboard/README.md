# FraudFlux Dashboard

A live transaction monitor for the trained Random Forest model (threshold 0.15).
Upload a CSV of unlabeled transactions, the API scores every row, and the UI streams
verdicts into the feed with running stats. Click any row to see why the model
reached that verdict (per-transaction SHAP feature contributions).

```
dashboard/
├── backend/          # FastAPI, serves models/random_forest_trees.npz
│   ├── main.py
│   ├── forest.py         # numpy Random Forest scoring + exact TreeSHAP
│   ├── export_model.py   # .pkl model + scaler -> .npz
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
| `FRAUDFLUX_MODEL_PATH` | `models/random_forest_trees.npz` |
| `FRAUDFLUX_THRESHOLD` | `0.15` |
| `FRAUDFLUX_CORS_ORIGINS` | `http://localhost:5173` |
| `FRAUDFLUX_MAX_ROWS` | `200000` |

Endpoints:

- `GET /api/health` -> model load status
- `GET /api/model` -> model name, threshold, expected columns, reported test metrics
- `POST /api/predict` (multipart `file`, optional `threshold` form field) -> per-row probability and verdict, summary
- `POST /api/explain` (multipart `file` holding one CSV row, `index`, optional `threshold`) -> SHAP contributions for that row: baseline probability, signed contribution and raw value per feature, plain-language summary

The API keeps no state between requests. The UI sends a row back to `/api/explain`
when it is clicked, and uploads large CSVs to `/api/predict` in chunks of about
3 MB to stay under serverless request limits.

The model runs without scikit-learn or shap: `forest.py` reimplements
`predict_proba` and path-dependent TreeSHAP in numpy on the arrays in
`models/random_forest_trees.npz`, which matches both libraries to floating-point
precision. After retraining, regenerate the file (needs scikit-learn and joblib):

```bash
python -m dashboard.backend.export_model
```

## Deploy to Vercel

The repo root is a Vercel project: `vercel.json` builds the frontend from
`dashboard/frontend` and serves the API from `api/index.py` (dependencies from
the root `requirements.txt`). Import the repo in Vercel with the root directory
left as the repo root and deploy; no settings or environment variables needed.

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
