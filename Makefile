PYTHON ?= python
PIP ?= $(PYTHON) -m pip
NSCA ?= nsca

.PHONY: install test lint api web-install web-dev web-build docker-up figures experiments real-data real-experiments cross-dataset policy-evaluation conformal-risk off-policy sensitivity-analyses negative-controls leakage-audit reporting-standard dataset-screening nmi-audit nmi-package power-study validate-study-platform generate-study-report real-figures audit-figures audit-render-safety audit-visual-text visual-feedback audit-claims paper supplement all private-paper

install:
	$(PIP) install -e .[dev]

test:
	pytest

lint:
	$(PYTHON) -m compileall src

api:
	uvicorn apps.api.main:app --reload --host 127.0.0.1 --port 8000

web-install:
	cd apps/web && npm install

web-dev:
	cd apps/web && npm run dev

web-build:
	cd apps/web && npm run build

docker-up:
	docker compose -f infra/docker-compose.yml up --build

figures:
	$(NSCA) generate-figures

experiments:
	$(NSCA) run-experiments

real-data:
	$(NSCA) download-real-data
	$(NSCA) prepare-real-data

real-experiments:
	$(NSCA) run-real-experiments

cross-dataset:
	$(NSCA) run-cross-dataset

policy-evaluation:
	$(NSCA) run-policy-evaluation

conformal-risk:
	$(NSCA) run-conformal-risk-control

off-policy:
	$(NSCA) run-off-policy-evaluation

sensitivity-analyses:
	$(NSCA) run-sensitivity-analyses

negative-controls:
	$(NSCA) run-negative-controls

leakage-audit:
	$(NSCA) run-leakage-audit

reporting-standard:
	$(NSCA) write-reporting-standard

dataset-screening:
	$(NSCA) screen-datasets

nmi-audit:
	$(NSCA) nmi-submission-audit

nmi-package:
	$(PYTHON) scripts/make_nmi_submission.py

power-study:
	$(NSCA) power-study

validate-study-platform:
	$(NSCA) validate-study-platform

generate-study-report:
	$(NSCA) generate-study-report

real-figures:
	$(NSCA) generate-real-figures

audit-figures:
	$(NSCA) audit-figures

audit-render-safety:
	$(NSCA) audit-render-safety

audit-visual-text:
	$(NSCA) audit-visual-text

visual-feedback:
	$(NSCA) visual-feedback-loop

audit-claims:
	$(NSCA) audit-claims

paper:
	$(NSCA) build-paper

supplement:
	$(NSCA) build-supplement

all: install real-data real-experiments cross-dataset policy-evaluation conformal-risk off-policy sensitivity-analyses negative-controls leakage-audit dataset-screening reporting-standard power-study validate-study-platform generate-study-report test lint

private-paper: real-figures audit-figures audit-render-safety audit-visual-text visual-feedback audit-claims nmi-audit paper supplement
