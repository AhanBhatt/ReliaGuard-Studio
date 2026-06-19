# Public Release Checklist

Use this before publishing the repository on GitHub.

- Confirm `paper/`, `artifacts/`, `.private/` and raw data remain ignored.
- Run `git status --ignored` and verify no manuscript, raw dataset, generated figure, source-data table or submission-audit file is staged.
- Run `pytest`.
- Run `python -m compileall src`.
- Replace the placeholder repository URL in `CITATION.cff`.
- Choose and add a real open-source license before public release. The current citation file marks the license as `NOASSERTION` because the repository owner has not selected one.
- Do not force-add ignored files unless intentionally publishing a separate paper artifact.
- Re-check third-party dataset licenses before redistributing any downloaded data.
