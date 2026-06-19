import { Badge, ButtonLink, Card, InstrumentCard, Metric, PageShell } from "@/components/ui";

const datasets = [
  ["HAIID", "35.7k", "advice-taking decisions"],
  ["CHI 2023 DKE", "4.0k", "tutorial and XAI interventions"],
  ["ConvXAI", "3.1k", "explanation interfaces"],
  ["Pardos/Bhandari", "274", "short-term tutoring gain"],
  ["FLoRA IPS", "275", "GenAI process traces"]
];

const outputs = [
  "Reliance-state probabilities",
  "Calibrated harmful-risk score",
  "Symbolic rule trace",
  "Counterfactual suggestion",
  "Selective-gating action"
];

export default function HomePage() {
  return (
    <PageShell eyebrow="Production AI evaluation platform" title="Average accuracy is a blunt instrument. ReliaGuard shows the hidden reliance failures.">
      <div className="grid gap-6 lg:grid-cols-[1.05fr_0.95fr]">
        <InstrumentCard className="instrument-dark min-h-[520px] text-paper">
          <div className="relative z-10">
            <div className="flex flex-wrap gap-2">
              <Badge tone="risk">Overreliance</Badge>
              <Badge tone="warn">Underreliance</Badge>
              <Badge tone="info">Calibration</Badge>
              <Badge tone="safe">Selective gating</Badge>
            </div>
            <p className="mt-8 max-w-3xl text-2xl font-bold leading-9 text-bone">
              AI systems can raise average performance while quietly increasing the wrong kind of dependence. ReliaGuard Studio turns each human-AI interaction into a risk-aware state transition: initial judgment, confidence, advice, final judgment, outcome and context.
            </p>
            <div className="mt-8 grid gap-3 md:grid-cols-2">
              {outputs.map((output, index) => (
                <div key={output} className="rounded-[1.4rem] border border-paper/20 bg-bone/12 p-4 shadow-[inset_0_1px_0_rgba(255,248,234,0.14)]">
                  <div className="text-xs font-black uppercase tracking-[0.22em] text-amber">Output {index + 1}</div>
                  <div className="mt-2 text-lg font-black text-bone">{output}</div>
                </div>
              ))}
            </div>
            <div className="mt-8 flex flex-wrap gap-3">
              <ButtonLink href="/simulator">Try simulator</ButtonLink>
              <ButtonLink href="/gating">Open gating dashboard</ButtonLink>
            </div>
          </div>
        </InstrumentCard>

        <div className="grid gap-6">
          <Card>
            <div className="grid gap-4 sm:grid-cols-3">
              <Metric label="Public records" value="43,263" />
              <Metric label="People represented" value="2,229" />
              <Metric label="Datasets" value="5" note="Decision, tutoring and GenAI traces" />
            </div>
          </Card>
          <Card>
            <div className="flex items-start justify-between gap-4">
              <div>
                <p className="text-xs font-black uppercase tracking-[0.22em] text-coral">Evidence base</p>
                <h2 className="mt-2 font-display text-3xl font-black tracking-[-0.035em]">Five public datasets, one product lens.</h2>
              </div>
              <span className="rounded-full border border-ink/15 bg-bone px-3 py-1 text-xs font-black uppercase tracking-[0.14em] text-ink/75">Real data</span>
            </div>
            <div className="mt-5 space-y-3">
              {datasets.map(([name, size, note]) => (
                <div key={name} className="grid grid-cols-[5rem_1fr] items-center gap-4 rounded-[1.25rem] bg-bone/70 p-3">
                  <div className="font-display text-2xl font-black text-coral">{size}</div>
                  <div>
                    <div className="font-black">{name}</div>
                    <div className="text-sm font-semibold text-ink/74">{note}</div>
                  </div>
                </div>
              ))}
            </div>
          </Card>
        </div>
      </div>

      <div className="mt-6 grid gap-6 md:grid-cols-3">
        <Card>
          <Badge tone="info">API-first</Badge>
          <h3 className="mt-4 text-2xl font-black tracking-[-0.025em]">FastAPI serves model outputs.</h3>
          <p className="mt-3 text-ink/70">The frontend calls a backend for predictions, explanations, thresholds, ablations, dataset summaries and model-card metadata.</p>
        </Card>
        <Card>
          <Badge tone="risk">Interpretable risk</Badge>
          <h3 className="mt-4 text-2xl font-black tracking-[-0.025em]">Rules make the warning actionable.</h3>
          <p className="mt-3 text-ink/70">A flagged case comes with active rules and a counterfactual such as “compare evidence before adopting wrong advice.”</p>
        </Card>
        <Card>
          <Badge tone="safe">Honest boundary</Badge>
          <h3 className="mt-4 text-2xl font-black tracking-[-0.025em]">Evaluation, not diagnosis.</h3>
          <p className="mt-3 text-ink/70">ReliaGuard supports AI evaluation and prospective intervention design, not clinical claims or causal proof of behavior change.</p>
        </Card>
      </div>
    </PageShell>
  );
}
