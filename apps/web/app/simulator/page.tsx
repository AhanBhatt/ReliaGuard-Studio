"use client";

import { useState } from "react";
import { Badge, Card, InstrumentCard, PageShell, ValueBar } from "@/components/ui";
import { postRelianceCase, type RelianceCase } from "@/lib/api";
import { displayLabel, percent } from "@/lib/format";

const initialCase: RelianceCase = {
  initial_answer: "A",
  initial_confidence: 0.82,
  ai_advice: "B",
  final_answer: "B",
  ground_truth: "A",
  task_context: "loan decision support",
  advice_source: "AI",
  model_confidence: 0.78
};

const fields: Array<[keyof RelianceCase, string, string]> = [
  ["initial_answer", "Initial answer", "Human first judgment"],
  ["ai_advice", "AI advice", "System recommendation"],
  ["final_answer", "Final answer", "Human final judgment"],
  ["ground_truth", "Ground truth", "Known correct outcome"],
  ["task_context", "Task context", "Domain or decision setting"],
  ["advice_source", "Advice source", "AI, human, dashboard, tutor"]
];

export default function SimulatorPage() {
  const [form, setForm] = useState<RelianceCase>(initialCase);
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  async function run() {
    setLoading(true);
    try {
      const prediction = await postRelianceCase("/predict-reliance", form);
      const explanation = await postRelianceCase("/explain-case", form);
      setResult({prediction, explanation});
    } finally {
      setLoading(false);
    }
  }

  function update<K extends keyof RelianceCase>(key: K, value: RelianceCase[K]) {
    setForm({...form, [key]: value});
  }

  return (
    <PageShell eyebrow="Interactive case simulator" title="Score one human-AI decision and inspect the rule trace.">
      <div className="grid gap-6 lg:grid-cols-[0.92fr_1.08fr]">
        <Card>
          <div className="flex items-start justify-between gap-4">
            <div>
              <Badge tone="info">Case input</Badge>
              <h2 className="mt-4 text-3xl font-black tracking-[-0.035em]">Decision tuple</h2>
              <p className="mt-2 text-sm font-semibold leading-6 text-ink/78">Enter the minimum observable sequence: initial judgment, advice, final judgment, outcome and context.</p>
            </div>
            <button onClick={() => setForm(initialCase)} className="rounded-full border border-ink/15 px-3 py-2 text-xs font-black uppercase tracking-[0.14em] text-ink/75 hover:bg-ink/10">
              Reset
            </button>
          </div>
          <div className="mt-6 grid gap-4">
            {fields.map(([key, label, helper]) => (
              <label key={key} className="block">
                <span className="flex items-center justify-between gap-3">
                  <span className="text-sm font-black text-ink/78">{label}</span>
                  <span className="text-xs font-bold text-ink/68">{helper}</span>
                </span>
                <input className="mt-2 w-full rounded-[1.15rem] border border-ink/12 bg-bone/80 px-4 py-3 font-bold text-ink shadow-[inset_0_1px_0_rgba(255,255,255,0.65)]" value={String(form[key])} onChange={(event) => update(key, event.target.value as never)} />
              </label>
            ))}
            <label className="text-sm font-black text-ink/78">
              Initial confidence <span className="text-coral">{form.initial_confidence.toFixed(2)}</span>
              <input type="range" min="0" max="1" step="0.01" value={form.initial_confidence} onChange={(event) => update("initial_confidence", Number(event.target.value))} className="range-control mt-2 w-full" />
            </label>
            <label className="text-sm font-black text-ink/78">
              Model confidence <span className="text-sky">{form.model_confidence.toFixed(2)}</span>
              <input type="range" min="0" max="1" step="0.01" value={form.model_confidence} onChange={(event) => update("model_confidence", Number(event.target.value))} className="range-control mt-2 w-full" />
            </label>
            <button onClick={run} className="mt-2 rounded-full bg-ink px-5 py-4 text-sm font-black uppercase tracking-[0.16em] text-paper shadow-soft transition hover:-translate-y-0.5 hover:bg-coral">
              {loading ? "Scoring case..." : "Predict reliance state"}
            </button>
          </div>
        </Card>

        {result ? (
          <InstrumentCard className="instrument-dark text-paper">
            <div className="relative z-10">
              <div className="flex flex-wrap gap-2">
                <Badge tone="risk">Risk score</Badge>
                <Badge tone="info">Rule-grounded</Badge>
              </div>
              <p className="mt-7 text-xs font-black uppercase tracking-[0.24em] text-coral">Predicted state</p>
              <h2 className="mt-2 font-display text-5xl font-black leading-none tracking-[-0.055em]">{displayLabel(result.prediction.state)}</h2>
              <div className="mt-7 grid gap-4 md:grid-cols-2">
                <div className="rounded-[1.5rem] bg-paper/10 p-5">
                  <div className="text-xs font-black uppercase tracking-[0.18em] text-paper/90">Harmful reliance risk</div>
                  <div className="mt-2 font-display text-5xl font-black text-coral">{percent(result.prediction.harmful_reliance_risk)}</div>
                  <ValueBar value={result.prediction.harmful_reliance_risk} tone="risk" />
                </div>
                <div className="rounded-[1.5rem] bg-paper/10 p-5">
                  <div className="text-xs font-black uppercase tracking-[0.18em] text-paper/90">Uncertainty</div>
                  <div className="mt-2 font-display text-5xl font-black text-signal">{percent(result.prediction.uncertainty)}</div>
                  <ValueBar value={result.prediction.uncertainty} tone="info" />
                </div>
              </div>

              <div className="mt-6 rounded-[1.75rem] border border-paper/12 bg-paper/10 p-5">
                <h3 className="text-xl font-black">Explanation card</h3>
                <p className="mt-3 font-semibold text-paper/90">{result.explanation.plain_language_summary}</p>
                <div className="mt-4 flex flex-wrap gap-2">
                  {result.prediction.active_rules.map((rule: string) => (
                    <span key={rule} className="rounded-full bg-paper/15 px-3 py-1 text-xs font-black uppercase tracking-[0.12em] text-paper/90">{displayLabel(rule)}</span>
                  ))}
                </div>
              </div>
              <div className="mt-4 grid gap-4 md:grid-cols-2">
                <div className="rounded-[1.5rem] bg-bone p-5 text-ink">
                  <h3 className="font-black">Counterfactual</h3>
                  <p className="mt-2 text-sm leading-6 text-ink/70">{result.prediction.counterfactual}</p>
                </div>
                <div className="rounded-[1.5rem] bg-coral p-5 text-paper">
                  <h3 className="font-black">Recommended action</h3>
                  <p className="mt-2 text-lg font-black">{displayLabel(result.prediction.recommended_action)}</p>
                </div>
              </div>
            </div>
          </InstrumentCard>
        ) : (
          <InstrumentCard className="instrument-dark text-paper">
            <div className="relative z-10 flex min-h-[520px] flex-col justify-between">
              <div>
                <Badge tone="warn">Awaiting case</Badge>
                <h2 className="mt-6 font-display text-5xl font-black leading-none tracking-[-0.05em] text-bone">A reliance trace will appear here.</h2>
                <p className="mt-5 max-w-lg text-lg font-bold leading-8 text-bone">Run the simulator to get calibrated risk, uncertainty, active rules, a counterfactual and an action recommendation.</p>
              </div>
              <div className="rounded-[1.75rem] border border-paper/20 bg-bone/12 p-5">
                <div className="text-xs font-black uppercase tracking-[0.2em] text-coral">Example warning</div>
                <p className="mt-3 text-base font-bold leading-7 text-bone">“The user was initially correct, advice was wrong, and the final answer moved toward the advice.”</p>
              </div>
            </div>
          </InstrumentCard>
        )}
      </div>
    </PageShell>
  );
}
