# Digital Twin demo copy

The twin (a forecast, a low-sugar alert and reasons for a doctor) lives in a separate demo copy of SmartPoli. The live app used by
family patients never shows it: every twin endpoint answers 404 unless `SMARTPOLI_TWIN_ENABLED=1`.

## What the demo copy needs
| Setting | Value |
|---|---|
| `SMARTPOLI_TWIN_ENABLED` | `1` (turns the twin on: `/twin/info`, `/twin/worklist`, `/patients/{id}/ml`, review actions, the Digital Twin menu item) |
| `SMARTPOLI_DEMO_LOGIN` | `1` (login page shows the "Doctor demo" card with one-tap sign-in and the credentials) |
| `SMARTPOLI_DEMO_SEED` | `1` (on start-up: the demo accounts from `seed.py`, then the two twin demo patients linked to `doctor@smartpoli.demo`; safe to repeat) |
| `SMARTPOLI_DATABASE_URL` | its OWN database (never the live one) |
| `SMARTPOLI_JWT_SECRET` | any new random secret |

Demo doctor: `doctor@smartpoli.demo` / `demo1234` (Dr. Rao).

## Demo patient data (not in the public repo)
`backend/twin_demo_data/demo_a.json` (ShanghaiT2DM recording 2094, CC BY 4.0, committed with attribution; see backend/twin_demo_data/README.md) and `demo_b.json`
(CGMacros participant 38, man 52, with heart rate) hold anonymised public research data, so they are git-ignored. Make them with
`python ml/export_demo_data.py` (see that file) and, for the private demo branch, add them with `git add -f`.
Without the files the app still starts; the Twin tab is just empty.

## Deploy
Create a second Render service from a PRIVATE repo/branch (for example `demo-twin`) with the settings above; it uses the same
`render.yaml`/`backend/Dockerfile` shape as the live service. `xgboost-cpu` (small CPU-only build) is in `backend/requirements.txt`;
models load on the first twin request (about 11 MB of JSON in `backend/twin_models/`). If the free plan runs out of memory, use a larger plan for the demo only.

## Run it locally
```
cd backend
SMARTPOLI_TWIN_ENABLED=1 SMARTPOLI_DEMO_LOGIN=1 SMARTPOLI_DEMO_SEED=1 SMARTPOLI_DATABASE_URL=sqlite:///./demo_local.db \
SMARTPOLI_DISABLE_SCHEDULER=1 SMARTPOLI_JWT_SECRET=local-secret uvicorn main:app --port 8010
```

## Keeping training and the app in step
`backend/twin_features.py` (plain Python) must equal `ml/features.py` (training). `tests/test_twin_features.py` compares them on a real
recording (skipped when the data is not on the machine). Change one, change both, rerun that test.

## Update (7 Oct): what is in the repo
`backend/twin_demo_data/demo_a.json` (ShanghaiT2DM, CC BY 4.0) is committed WITH attribution (README beside it, and a credit line on the Twin tab), so the demo patient A works from the repo alone.
`demo_b.json` (CGMacros, share-alike / non-commercial) stays out of the public repo; without it the Twin tab simply shows patient A. Never describe either patient as made up: they are anonymised real research data, renamed "Demo patient A / B".
