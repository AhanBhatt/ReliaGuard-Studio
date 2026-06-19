# Infrastructure

Local product stack:

```powershell
docker compose -f infra/docker-compose.yml up --build
```

Services:

- Web: `http://localhost:3000`
- API: `http://localhost:8000`
- API docs: `http://localhost:8000/docs`

The compose file mounts ignored `artifacts/` locally so public datasets and model outputs are not committed to Git.
