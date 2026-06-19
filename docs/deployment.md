# Deployment Guide

## Local API

```powershell
python -m pip install -e ".[dev]"
uvicorn apps.api.main:app --reload --host 127.0.0.1 --port 8000
```

## Local web app

```powershell
cd apps/web
npm install
npm run dev
```

## Full stack

```powershell
docker compose -f infra/docker-compose.yml up --build
```

## Required review before public deployment

- Confirm repository license.
- Confirm data-source licenses.
- Do not deploy raw participant-level data.
- Keep `paper/`, `artifacts/` and `.private/` out of public source control.
- Add authentication before exposing private dashboards or runtime databases.
