# Publishing ReliaGuard Studio As A New GitHub Repository

This project is intended to be published as a new repository named:

```text
ReliaGuard-Studio
```

Recommended GitHub URL:

```text
https://github.com/AhanBhatt/ReliaGuard-Studio
```

The repository display title should be:

```text
ReliaGuard Studio
```

## Important Boundary

Do not publish this over any previous prototype repository. This folder has been reinitialized as a new local Git repository for ReliaGuard Studio.

## Before Pushing

Verify ignored private materials are not staged:

```powershell
git status --ignored
```

The following should remain ignored:

- `paper/`
- `artifacts/`
- `.private/`
- `CLAIMS_CHECKLIST.md`
- generated audit files
- generated manuscript/source-data artifacts
- local caches such as `.next/`, `node_modules/`, `__pycache__/`, `.runtime_logs/`

## First Push

Create an empty GitHub repository named `ReliaGuard-Studio`, then run:

```powershell
cd "D:\Projects\ReliaGuard Studio"
git add .
git commit -m "Initial ReliaGuard Studio product release"
git branch -M main
git remote -v
git push -u origin main
```

If `origin` is missing, add it:

```powershell
git remote add origin https://github.com/AhanBhatt/ReliaGuard-Studio.git
```

If the remote URL is wrong, reset it:

```powershell
git remote set-url origin https://github.com/AhanBhatt/ReliaGuard-Studio.git
```

## Suggested GitHub Description

```text
Production-style AI evaluation platform for detecting overreliance, underreliance, and unsafe human-AI decision behavior.
```

## Suggested Topics

```text
human-ai-interaction, ai-safety, machine-learning, fastapi, nextjs, model-evaluation, xai, calibration, conformal-prediction
```
