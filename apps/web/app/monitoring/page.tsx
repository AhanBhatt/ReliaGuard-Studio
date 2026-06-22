"use client";

import { useEffect, useState } from "react";
import { Badge, Card, PageShell, ValueBar } from "@/components/ui";
import { getJson } from "@/lib/api";
import { displayLabel, percent } from "@/lib/format";

export default function MonitoringPage() {
  const [projectId, setProjectId] = useState("");
  const [summary, setSummary] = useState<any>({});

  async function refresh() {
    const path = projectId ? `/v1/monitoring?project_id=${encodeURIComponent(projectId)}` : "/v1/monitoring";
    const json = await getJson(path);
    setSummary(json);
  }

  useEffect(() => {
    refresh().catch(() => {});
  }, []);

  const stateRows = Object.entries(summary.state_counts ?? {});
  const actionRows = Object.entries(summary.action_counts ?? {});

  return (
    <PageShell eyebrow="Shadow and guardrail monitoring" title="Track reliance risk over time, by context, and by reviewer feedback.">
      <Card>
        <div className="flex flex-wrap items-end justify-between gap-4">
          <div>
            <Badge tone="info">Production observability</Badge>
            <h2 className="mt-4 text-3xl font-black tracking-[-0.035em]">{summary.events ?? 0} logged events</h2>
            <p className="mt-2 font-semibold leading-7 text-ink/72">Use audit mode first, stream in shadow mode next, and enable guardrail mode only when thresholds and burden are acceptable.</p>
          </div>
          <label className="grid gap-2 text-xs font-black uppercase tracking-[0.14em] text-ink/70">
            Project filter
            <input value={projectId} onChange={(event) => setProjectId(event.target.value)} onBlur={refresh} placeholder="optional project_id" className="rounded-full border border-ink/12 bg-bone px-4 py-2 normal-case tracking-normal text-ink" />
          </label>
        </div>
      </Card>

      <div className="mt-6 grid gap-6 lg:grid-cols-3">
        <Card>
          <Badge tone="risk">Reliance states</Badge>
          <div className="mt-5 space-y-4">
            {stateRows.map(([state, count]) => {
              const value = Number(count) / Math.max(1, Number(summary.events ?? 0));
              return (
                <div key={state}>
                  <div className="mb-2 flex justify-between text-xs font-black text-ink/75"><span>{displayLabel(state)}</span><span>{percent(value, 1)}</span></div>
                  <ValueBar value={value} tone={String(state).includes("harmful") ? "risk" : "info"} />
                </div>
              );
            })}
            {!stateRows.length ? <p className="font-semibold text-ink/72">No state telemetry yet.</p> : null}
          </div>
        </Card>
        <Card>
          <Badge tone="warn">Actions</Badge>
          <div className="mt-5 space-y-4">
            {actionRows.map(([action, count]) => {
              const value = Number(count) / Math.max(1, Number(summary.events ?? 0));
              return (
                <div key={action}>
                  <div className="mb-2 flex justify-between text-xs font-black text-ink/75"><span>{displayLabel(action)}</span><span>{percent(value, 1)}</span></div>
                  <ValueBar value={value} tone="info" />
                </div>
              );
            })}
            {!actionRows.length ? <p className="font-semibold text-ink/72">No action telemetry yet.</p> : null}
          </div>
        </Card>
        <Card>
          <Badge tone="safe">Reviewer feedback</Badge>
          <div className="mt-5 space-y-3">
            {Object.entries(summary.review_feedback ?? {}).map(([label, count]) => (
              <div key={label} className="flex justify-between rounded-[1rem] bg-bone/72 p-3 font-bold text-ink/76">
                <span>{displayLabel(label)}</span>
                <span>{String(count)}</span>
              </div>
            ))}
            {!Object.keys(summary.review_feedback ?? {}).length ? <p className="font-semibold text-ink/72">No review labels yet.</p> : null}
          </div>
        </Card>
      </div>

      <div className="mt-6 grid gap-4 lg:grid-cols-2">
        {(summary.by_context ?? []).map((row: any) => (
          <Card key={row.context}>
            <div className="flex justify-between gap-4">
              <div>
                <div className="text-xs font-black uppercase tracking-[0.16em] text-coral">Context</div>
                <h2 className="mt-2 text-2xl font-black">{displayLabel(row.context)}</h2>
              </div>
              <div className="text-right font-display text-4xl font-black text-coral">{percent(row.mean_risk)}</div>
            </div>
            <div className="mt-5 grid gap-4 md:grid-cols-2">
              <div><div className="mb-2 text-xs font-black text-ink/75">Overreliance {percent(row.overreliance_rate, 1)}</div><ValueBar value={row.overreliance_rate} tone="risk" /></div>
              <div><div className="mb-2 text-xs font-black text-ink/75">Underreliance {percent(row.underreliance_rate, 1)}</div><ValueBar value={row.underreliance_rate} tone="info" /></div>
            </div>
          </Card>
        ))}
      </div>
    </PageShell>
  );
}
