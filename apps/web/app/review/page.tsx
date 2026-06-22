"use client";

import { useEffect, useState } from "react";
import { Badge, Card, PageShell } from "@/components/ui";
import { getJson, postJson } from "@/lib/api";
import { displayLabel, percent } from "@/lib/format";

const labels = [
  "true_harmful_reliance",
  "false_positive",
  "uncertain",
  "bad_ground_truth",
  "intervention_worked",
  "intervention_too_burdensome"
];

export default function ReviewQueuePage() {
  const [cases, setCases] = useState<any[]>([]);
  const [projectId, setProjectId] = useState("");

  async function refresh() {
    const path = projectId ? `/v1/review-queue?project_id=${encodeURIComponent(projectId)}` : "/v1/review-queue";
    const json = await getJson(path);
    setCases(json.cases ?? []);
  }

  useEffect(() => {
    refresh().catch(() => {});
  }, []);

  async function labelCase(row: any, label: string) {
    await postJson("/v1/review-queue/label", {
      project_id: row.project_id,
      case_id: row.case_id,
      label,
      reviewer: "local_reviewer",
      note: "Updated from review queue"
    });
    await refresh();
  }

  return (
    <PageShell eyebrow="Case review queue" title="Close the loop: label flagged cases and tune policies with real feedback.">
      <Card>
        <div className="flex flex-wrap items-end justify-between gap-4">
          <div>
            <Badge tone="risk">Flagged cases</Badge>
            <h2 className="mt-4 text-3xl font-black tracking-[-0.035em]">{cases.length} cases waiting or labelled</h2>
            <p className="mt-2 font-semibold leading-7 text-ink/72">Reviewers can mark true harmful reliance, false positives, uncertain cases, bad ground truth, and intervention burden outcomes.</p>
          </div>
          <label className="grid gap-2 text-xs font-black uppercase tracking-[0.14em] text-ink/70">
            Project filter
            <input value={projectId} onChange={(event) => setProjectId(event.target.value)} onBlur={refresh} placeholder="optional project_id" className="rounded-full border border-ink/12 bg-bone px-4 py-2 normal-case tracking-normal text-ink" />
          </label>
        </div>
      </Card>
      <div className="mt-6 grid gap-4">
        {cases.map((row) => (
          <Card key={row.case_id}>
            <div className="grid gap-5 lg:grid-cols-[1.2fr_0.8fr_1fr]">
              <div>
                <div className="flex flex-wrap gap-2">
                  <Badge tone={row.state?.includes("harmful") ? "risk" : "warn"}>{displayLabel(row.state)}</Badge>
                  <span className="rounded-full bg-bone px-3 py-1 font-mono text-xs font-black text-ink/70">{row.case_id}</span>
                </div>
                <h2 className="mt-4 text-2xl font-black tracking-[-0.025em]">{row.task_id}</h2>
                <p className="mt-2 text-sm font-semibold leading-6 text-ink/72">
                  Initial <b>{row.excerpt?.initial_answer}</b> · Advice <b>{row.excerpt?.ai_advice}</b> · Final <b>{row.excerpt?.final_answer}</b> · Truth <b>{row.excerpt?.ground_truth}</b>
                </p>
              </div>
              <div>
                <div className="mb-2 flex justify-between text-xs font-black uppercase tracking-[0.12em] text-ink/70"><span>Risk</span><span>{percent(row.risk)}</span></div>
                <div className="h-3 overflow-hidden rounded-full bg-ink/10"><div className="h-full rounded-full bg-coral" style={{width: `${Number(row.risk ?? 0) * 100}%`}} /></div>
                <div className="mt-4 rounded-[1rem] bg-bone/72 p-3 text-sm font-bold text-ink/74">Action: {displayLabel(row.action)}</div>
                <div className="mt-2 rounded-[1rem] bg-bone/72 p-3 text-sm font-bold text-ink/74">Decision: {displayLabel(row.reviewer_decision)}</div>
              </div>
              <div className="grid grid-cols-2 gap-2">
                {labels.map((label) => (
                  <button key={label} onClick={() => labelCase(row, label)} className="rounded-[1rem] border border-ink/10 bg-bone/75 px-3 py-2 text-left text-xs font-black uppercase leading-5 tracking-[0.1em] text-ink/76 transition hover:bg-ink hover:text-paper">
                    {displayLabel(label)}
                  </button>
                ))}
              </div>
            </div>
          </Card>
        ))}
        {!cases.length ? (
          <Card>
            <h2 className="text-2xl font-black">No flagged cases yet.</h2>
            <p className="mt-2 font-semibold text-ink/72">Upload logs, stream SDK events, or call the guardrail endpoint to populate the review queue.</p>
          </Card>
        ) : null}
      </div>
    </PageShell>
  );
}
