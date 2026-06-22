# Quickstart: Audit Your First Dataset In 5 Minutes

ReliaGuard Studio is a local-first human-AI reliance observability platform. The fastest path is to upload historical logs and generate an audit report.

## 1. Start the app

```powershell
.\run_app.ps1
```

Open the printed web URL, usually `http://127.0.0.1:3000`.

## 2. Open Upload Logs

Go to `Upload Logs`.

You can load the built-in demo or upload:

- CSV
- JSON
- JSONL / NDJSON

Parquet is part of the ingestion contract, but the browser demo keeps parsing dependency-free. For production Parquet ingestion, add a backend multipart adapter with parquet support.

## 3. Map columns

Map your columns to the canonical schema:

| ReliaGuard field | Common source columns |
| --- | --- |
| `user_id` | `agent_id`, `learner_id`, `reviewer_id` |
| `task_id` | `ticket_id`, `question_id`, `case_id` |
| `initial_answer` | `human_first_decision` |
| `initial_confidence` | `confidence_before_ai` |
| `ai_advice` | `model_recommendation` |
| `ai_confidence` | `model_confidence` |
| `final_answer` | `human_final_decision` |
| `ground_truth` | `true_label`, `QA outcome`, `test_result` |
| `context` | `domain`, `task_type`, `model_name` |

## 4. Generate the audit

Click `Generate audit report`.

The report shows:

- scored records;
- missing fields;
- analyses possible;
- reliance-state distribution;
- overreliance rate;
- underreliance rate;
- mean harmful-risk score;
- highest-risk cases;
- threshold recommendation;
- limitations.

## 5. Review and replay

Open `Review Queue` to label flagged cases.

Open `Replay Logs` to ask what would have happened at a different alpha or policy.

## Evidence boundary

Uploaded logs support observational audit and replay. They do not prove that interventions causally change behavior.
