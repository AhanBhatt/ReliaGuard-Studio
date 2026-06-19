# ReliaGuard Studio API

FastAPI service layer for the ReliaGuard Studio product interface.

Run locally from the repository root:

```powershell
uvicorn apps.api.main:app --reload --host 127.0.0.1 --port 8000
```

Primary endpoints:

- `POST /predict-reliance`
- `POST /explain-case`
- `GET /conformal-threshold?alpha=0.10`
- `GET /simulate-policy`
- `GET /run-ablation`
- `GET /datasets`
- `GET /model-card`
- `GET /evaluation-lab`

The service imports the production API from `reliaguard_studio.api.main` so the public Python package remains the source of truth.
