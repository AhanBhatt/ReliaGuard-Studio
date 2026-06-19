# Experiments

The authoritative experiment logic lives in the package under `src/reliaguard_studio/`. This folder contains thin entrypoints that call the packaged pipeline so that experiments can be run either through the CLI or directly as scripts.

## Scripts

- `python experiments/run_air_bench.py`
- `python experiments/generate_assets.py`

The preferred interface remains:

```bash
nsca run-experiments
nsca generate-figures
```
