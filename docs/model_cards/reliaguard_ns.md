# Model Card: ReliaGuard-NS

## Intended use

ReliaGuard-NS supports AI evaluation by estimating harmful reliance risk, surfacing symbolic rule traces, and comparing selective-gating policies across public human-AI datasets.

## Inputs

- Initial human judgment.
- Initial confidence or available confidence proxy.
- AI/human advice or support.
- Final human judgment.
- Ground truth.
- Task/domain context.

## Outputs

- Reliance-state probabilities.
- Harmful-reliance risk.
- Active symbolic rules.
- Counterfactual explanation.
- Candidate action for prospective validation.

## Evidence base

The current public-data package evaluates 43,263 records from 2,229 participants/students across HAIID, CHI 2023 DKE, ConvXAI, Pardos/Bhandari, and FLoRA IPS.

## Limitations

- Not a universal raw-prediction winner.
- No completed prospective randomized intervention trial.
- No clinical, diagnostic, delayed-recall, transfer, or long-term-learning claim.
- Conformal outputs require exchangeability assumptions.
