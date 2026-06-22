"use client";

import { useState } from "react";
import { Badge, Card, PageShell, ValueBar } from "@/components/ui";
import { postJson } from "@/lib/api";
import { displayLabel, percent } from "@/lib/format";

const policies = ["confidence_threshold", "symbolic_rule", "reliaguard_ns"];

export default function ReplayPage() {
  const [projectId, setProjectId] = useState("customer-support-copilot");
  const [alpha, setAlpha] = useState(0.10);
  const [policy, setPolicy] = useState("reliaguard_ns");
  const [result, setResult] = useState<any>(null);

  async function runReplay() {
    const json = await postJson("/v1/replay", {project_id: projectId, alpha, policy});
    setResult(json);
  }

  return (
    <PageShell eyebrow="Replay logs" title="Ask what would have happened under a different guardrail threshold.">
      <div className="grid gap-6 lg:grid-cols-[0.85fr_1.15fr]">
        <Card>
          <Badge tone="warn">Historical replay</Badge>
          <h2 className="mt-4 text-3xl font-black tracking-[-0.035em]">Policy sandbox</h2>
          <p className="mt-3 font-semibold leading-7 text-ink/72">Replay uploaded or streamed logs under confidence thresholds, symbolic rules, or ReliaGuard-NS gating. This estimates flagging behavior, not causal behavior change.</p>
          <div className="mt-6 grid gap-4">
            <label className="grid gap-2 text-sm font-black uppercase tracking-[0.14em] text-ink/70">
              Project ID
              <input value={projectId} onChange={(event) => setProjectId(event.target.value)} className="rounded-[1rem] border border-ink/12 bg-bone/85 px-4 py-3 normal-case tracking-normal text-ink" />
            </label>
            <label className="grid gap-2 text-sm font-black uppercase tracking-[0.14em] text-ink/70">
              Policy
              <select value={policy} onChange={(event) => setPolicy(event.target.value)} className="rounded-[1rem] border border-ink/12 bg-bone/85 px-4 py-3 normal-case tracking-normal text-ink">
                {policies.map((item) => <option key={item} value={item}>{displayLabel(item)}</option>)}
              </select>
            </label>
            <label className="grid gap-2 text-sm font-black uppercase tracking-[0.14em] text-ink/70">
              Alpha {alpha.toFixed(2)}
              <input className="range-control" type="range" min="0.01" max="0.30" step="0.01" value={alpha} onChange={(event) => setAlpha(Number(event.target.value))} />
            </label>
            <button onClick={runReplay} className="rounded-full bg-ink px-5 py-3 text-sm font-black uppercase tracking-[0.14em] text-paper shadow-soft transition hover:bg-coral">Replay logs</button>
          </div>
        </Card>

        <Card>
          <Badge tone="safe">Replay output</Badge>
          {result ? (
            <div>
              <h2 className="mt-4 text-3xl font-black tracking-[-0.035em]">{result.flagged_cases} of {result.total_cases} cases flagged</h2>
              <div className="mt-6 grid gap-5 md:grid-cols-2">
                <div>
                  <div className="mb-2 flex justify-between text-xs font-black text-ink/75"><span>Harmful capture</span><span>{percent(result.harmful_capture)}</span></div>
                  <ValueBar value={result.harmful_capture} tone="safe" />
                </div>
                <div>
                  <div className="mb-2 flex justify-between text-xs font-black text-ink/75"><span>Intervention burden</span><span>{percent(result.intervention_burden)}</span></div>
                  <ValueBar value={result.intervention_burden} tone="info" />
                </div>
                <div className="rounded-[1rem] bg-bone/72 p-4 font-bold text-ink/76">Missed harmful<br /><b className="text-2xl text-coral">{result.missed_harmful}</b></div>
                <div className="rounded-[1rem] bg-bone/72 p-4 font-bold text-ink/76">Normal interrupted<br /><b className="text-2xl text-amber">{result.normal_cases_interrupted}</b></div>
              </div>
              <p className="mt-5 rounded-[1.25rem] bg-ink p-4 font-semibold leading-6 text-paper">{result.boundary}</p>
              <div className="mt-5 space-y-3">
                {(result.examples ?? []).map((row: any) => (
                  <div key={row.case_id} className="rounded-[1.2rem] border border-ink/10 bg-bone/70 p-3">
                    <div className="text-xs font-black uppercase tracking-[0.14em] text-coral">{row.task_id} · {displayLabel(row.state)}</div>
                    <div className="mt-1 text-sm font-semibold text-ink/72">{displayLabel(row.recommended_action)} · risk {percent(row.risk)}</div>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <p className="mt-4 font-semibold leading-7 text-ink/72">Run a replay after uploading logs or streaming events. The output will show harmful cases caught, missed harmful cases, normal cases interrupted, and policy burden.</p>
          )}
        </Card>
      </div>
    </PageShell>
  );
}
