"use client";

import { useEffect, useMemo, useState } from "react";
import { Badge, ButtonLink, Card, PageShell, ValueBar } from "@/components/ui";
import { getJson, postJson } from "@/lib/api";
import { displayLabel, percent } from "@/lib/format";

const fields = [
  ["user_id", "User ID", "agent_id, learner_id, reviewer_id"],
  ["task_id", "Task ID", "ticket_id, question_id, case_id"],
  ["initial_answer", "Initial answer", "human_first_decision"],
  ["initial_confidence", "Initial confidence", "confidence_before_ai"],
  ["ai_advice", "AI advice", "model_recommendation"],
  ["ai_confidence", "AI confidence", "model_confidence"],
  ["final_answer", "Final answer", "human_final_decision"],
  ["ground_truth", "Ground truth", "true_label, QA outcome, test result"],
  ["context", "Context", "domain, task type, model name"]
];

function parseCsv(text: string): Record<string, string>[] {
  const rows: string[][] = [];
  let cell = "";
  let row: string[] = [];
  let quoted = false;
  for (let index = 0; index < text.length; index += 1) {
    const char = text[index];
    const next = text[index + 1];
    if (char === '"' && quoted && next === '"') {
      cell += '"';
      index += 1;
    } else if (char === '"') {
      quoted = !quoted;
    } else if (char === "," && !quoted) {
      row.push(cell);
      cell = "";
    } else if ((char === "\n" || char === "\r") && !quoted) {
      if (char === "\r" && next === "\n") index += 1;
      row.push(cell);
      if (row.some((value) => value.trim() !== "")) rows.push(row);
      row = [];
      cell = "";
    } else {
      cell += char;
    }
  }
  if (cell || row.length) {
    row.push(cell);
    rows.push(row);
  }
  const headers = rows.shift()?.map((value) => value.trim()) ?? [];
  return rows.map((values) => Object.fromEntries(headers.map((header, index) => [header, values[index] ?? ""])));
}

function parseText(filename: string, text: string): {records: Record<string, unknown>[]; fileType: "csv" | "jsonl" | "json" | "manual"} {
  const lower = filename.toLowerCase();
  if (lower.endsWith(".jsonl") || lower.endsWith(".ndjson")) {
    return {fileType: "jsonl", records: text.split(/\r?\n/).filter(Boolean).map((line) => JSON.parse(line))};
  }
  if (lower.endsWith(".json")) {
    const parsed = JSON.parse(text);
    return {fileType: "json", records: Array.isArray(parsed) ? parsed : parsed.records ?? []};
  }
  return {fileType: "csv", records: parseCsv(text)};
}

export default function UploadLogsPage() {
  const [records, setRecords] = useState<Record<string, unknown>[]>([]);
  const [columns, setColumns] = useState<string[]>([]);
  const [mapping, setMapping] = useState<Record<string, string>>({});
  const [fileType, setFileType] = useState<"csv" | "jsonl" | "json" | "manual">("manual");
  const [sourceName, setSourceName] = useState("customer_support_logs");
  const [projectId, setProjectId] = useState("customer-support-copilot");
  const [audit, setAudit] = useState<any>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const missingFields = useMemo(() => fields.map(([field]) => field).filter((field) => !mapping[field]), [mapping]);

  async function loadDemo() {
    setError("");
    const demo = await getJson("/v1/demo/customer-support");
    const rows = demo.records ?? [];
    setRecords(rows);
    setColumns(Object.keys(rows[0] ?? {}));
    setMapping(demo.mapping ?? {});
    setSourceName("customer_support_demo");
    setProjectId("customer-support-copilot");
    setFileType("manual");
    setAudit(null);
  }

  useEffect(() => {
    loadDemo().catch(() => {});
  }, []);

  async function handleFile(file: File | null) {
    if (!file) return;
    setError("");
    if (file.name.toLowerCase().endsWith(".parquet")) {
      setError("Parquet is supported by the backend ingestion contract, but this browser demo parses CSV, JSON, and JSONL locally. Convert Parquet to CSV or add a backend multipart upload adapter for production.");
      return;
    }
    const text = await file.text();
    const parsed = parseText(file.name, text);
    const rows = parsed.records;
    setRecords(rows);
    setColumns(Object.keys(rows[0] ?? {}));
    setFileType(parsed.fileType);
    setSourceName(file.name.replace(/\.[^.]+$/, ""));
    setAudit(null);
    try {
      const preview = await postJson("/v1/ingest/preview", rows.slice(0, 50));
      setMapping(preview.suggested_mapping ?? {});
    } catch {
      setMapping({});
    }
  }

  async function runAudit() {
    setLoading(true);
    setError("");
    try {
      const result = await postJson("/v1/ingest/validate", {
        project_id: projectId,
        source_name: sourceName,
        records,
        mapping,
        file_type: fileType
      });
      setAudit(result);
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : "Audit failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <PageShell eyebrow="Audit mode" title="Bring your own human-AI logs and get a reliance audit.">
      <div className="grid gap-6 xl:grid-cols-[0.95fr_1.05fr]">
        <Card>
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div>
              <Badge tone="safe">Upload logs</Badge>
              <h2 className="mt-4 text-3xl font-black tracking-[-0.035em]">CSV, JSON, or JSONL ingestion</h2>
              <p className="mt-3 font-semibold leading-7 text-ink/78">
                ReliaGuard maps your columns into a canonical interaction schema, validates missing fields, and explains which analyses are possible before scoring.
              </p>
            </div>
            <button onClick={loadDemo} className="rounded-full border border-ink/15 bg-bone px-4 py-2 text-xs font-black uppercase tracking-[0.14em] text-ink hover:bg-ink hover:text-paper">
              Load demo
            </button>
          </div>
          <div className="mt-6 grid gap-4">
            <label className="grid gap-2">
              <span className="text-sm font-black uppercase tracking-[0.14em] text-ink/70">Project ID</span>
              <input value={projectId} onChange={(event) => setProjectId(event.target.value)} className="rounded-[1rem] border border-ink/12 bg-bone/85 px-4 py-3 font-bold text-ink" />
            </label>
            <label className="grid gap-2">
              <span className="text-sm font-black uppercase tracking-[0.14em] text-ink/70">Source name</span>
              <input value={sourceName} onChange={(event) => setSourceName(event.target.value)} className="rounded-[1rem] border border-ink/12 bg-bone/85 px-4 py-3 font-bold text-ink" />
            </label>
            <label className="rounded-[1.5rem] border border-dashed border-ink/25 bg-bone/60 p-5">
              <span className="block text-sm font-black uppercase tracking-[0.16em] text-coral">Choose file</span>
              <input type="file" accept=".csv,.json,.jsonl,.ndjson,.parquet" onChange={(event) => handleFile(event.target.files?.[0] ?? null)} className="mt-3 block w-full text-sm font-bold text-ink" />
              <span className="mt-3 block text-sm font-semibold leading-6 text-ink/72">Parquet support is part of the ingestion contract; this no-dependency browser demo parses CSV/JSON/JSONL directly.</span>
            </label>
          </div>
          {error ? <div className="mt-4 rounded-[1rem] bg-coral/10 p-4 font-bold text-coral">{error}</div> : null}
        </Card>

        <Card>
          <Badge tone="info">Column mapping</Badge>
          <h2 className="mt-4 text-3xl font-black tracking-[-0.035em]">Map your schema once.</h2>
          <div className="mt-6 space-y-3">
            {fields.map(([field, label, examples]) => (
              <div key={field} className="grid gap-3 rounded-[1.25rem] border border-ink/10 bg-bone/70 p-3 md:grid-cols-[1fr_1.15fr]">
                <div>
                  <div className="font-black text-ink">{label}</div>
                  <div className="text-xs font-semibold leading-5 text-ink/66">{examples}</div>
                </div>
                <select value={mapping[field] ?? ""} onChange={(event) => setMapping({...mapping, [field]: event.target.value})} className="rounded-[0.9rem] border border-ink/12 bg-paper px-3 py-2 font-bold text-ink">
                  <option value="">Not mapped</option>
                  {columns.map((column) => <option key={column} value={column}>{column}</option>)}
                </select>
              </div>
            ))}
          </div>
          <div className="mt-5 flex flex-wrap items-center justify-between gap-3">
            <div className="text-sm font-bold text-ink/72">{records.length.toLocaleString()} records loaded · {missingFields.length ? `${missingFields.length} fields missing` : "required fields mapped"}</div>
            <button disabled={!records.length || loading} onClick={runAudit} className="rounded-full bg-ink px-5 py-3 text-sm font-black uppercase tracking-[0.14em] text-paper shadow-soft transition hover:bg-coral disabled:opacity-45">
              {loading ? "Generating audit..." : "Generate audit report"}
            </button>
          </div>
        </Card>
      </div>

      {audit ? (
        <div className="mt-6 grid gap-6 lg:grid-cols-[0.8fr_1.2fr]">
          <Card>
            <Badge tone="risk">Audit report</Badge>
            <h2 className="mt-4 text-3xl font-black tracking-[-0.035em]">{audit.scored_records} scored interactions</h2>
            <div className="mt-5 grid gap-4 sm:grid-cols-3 lg:grid-cols-1">
              <div>
                <div className="mb-2 flex justify-between text-xs font-black text-ink/75"><span>Overreliance</span><span>{percent(audit.overreliance_rate, 1)}</span></div>
                <ValueBar value={audit.overreliance_rate} tone="risk" />
              </div>
              <div>
                <div className="mb-2 flex justify-between text-xs font-black text-ink/75"><span>Underreliance</span><span>{percent(audit.underreliance_rate, 1)}</span></div>
                <ValueBar value={audit.underreliance_rate} tone="info" />
              </div>
              <div>
                <div className="mb-2 flex justify-between text-xs font-black text-ink/75"><span>Mean harmful risk</span><span>{percent(audit.mean_risk, 1)}</span></div>
                <ValueBar value={audit.mean_risk} tone="risk" />
              </div>
            </div>
            <p className="mt-5 rounded-[1.25rem] bg-bone/70 p-4 text-sm font-semibold leading-6 text-ink/74">{audit.threshold_recommendation}</p>
            <div className="mt-4 flex flex-wrap gap-3">
              <ButtonLink href="/review">Open review queue</ButtonLink>
              <ButtonLink href="/replay">Replay thresholds</ButtonLink>
            </div>
          </Card>
          <Card>
            <Badge tone="safe">Highest-risk cases</Badge>
            <div className="mt-5 space-y-3">
              {(audit.highest_risk_cases ?? []).map((row: any) => (
                <div key={row.case_id} className="grid gap-3 rounded-[1.35rem] bg-bone/72 p-4 md:grid-cols-[1.2fr_0.8fr]">
                  <div>
                    <div className="text-xs font-black uppercase tracking-[0.16em] text-coral">{row.task_id ?? "case"} · {displayLabel(row.state)}</div>
                    <div className="mt-2 font-semibold text-ink/78">Initial: <b>{String(row.initial_answer)}</b> · Advice: <b>{String(row.ai_advice)}</b> · Final: <b>{String(row.final_answer)}</b> · Truth: <b>{String(row.ground_truth)}</b></div>
                  </div>
                  <div className="text-right">
                    <div className="font-display text-4xl font-black text-coral">{percent(row.risk)}</div>
                    <div className="text-xs font-black uppercase tracking-[0.12em] text-ink/62">{displayLabel(row.recommended_action)}</div>
                  </div>
                </div>
              ))}
            </div>
          </Card>
        </div>
      ) : null}
    </PageShell>
  );
}
