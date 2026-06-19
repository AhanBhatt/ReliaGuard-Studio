import { Badge, Card, PageShell } from "@/components/ui";

const steps = [
  ["Start the API", "Run uvicorn apps.api.main:app --reload --host 127.0.0.1 --port 8000 from the repository root."],
  ["Start the web app", "Run npm install and npm run dev inside apps/web, then open http://127.0.0.1:3000."],
  ["Try a case", "Use the simulator to enter initial answer, confidence, advice, final answer, ground truth and context."],
  ["Read the trace", "Inspect state, harmful-risk score, uncertainty, active rules, counterfactual and recommended action."],
  ["Tune alpha", "Open the gating dashboard and move alpha to inspect capture, missed harm and intervention burden."],
  ["Review evidence", "Use Evaluation, Policy and Data pages to understand model metrics, simulation limits and dataset provenance."]
];

const apiCalls = [
  "GET /datasets",
  "GET /model-card",
  "GET /conformal-threshold?alpha=0.20",
  "POST /predict-reliance",
  "POST /explain-case",
  "GET /simulate-policy"
];

export default function GuidePage() {
  return (
    <PageShell eyebrow="How to use ReliaGuard" title="A guided path from local setup to reliance-risk interpretation.">
      <div className="grid gap-6 lg:grid-cols-[1.1fr_0.9fr]">
        <Card>
          <Badge tone="safe">Demo sequence</Badge>
          <div className="mt-6 space-y-4">
            {steps.map(([title, body], index) => (
              <div key={title} className="grid grid-cols-[2.5rem_1fr] gap-4 rounded-[1.5rem] bg-bone/72 p-4">
                <div className="flex h-10 w-10 items-center justify-center rounded-full bg-coral font-black text-paper">{index + 1}</div>
                <div>
                  <h2 className="text-xl font-black">{title}</h2>
                  <p className="mt-1 font-semibold leading-6 text-ink/78">{body}</p>
                </div>
              </div>
            ))}
          </div>
        </Card>
        <div className="grid gap-6">
          <Card>
            <Badge tone="info">API surface</Badge>
            <h2 className="mt-4 text-3xl font-black tracking-[-0.035em]">Use the backend directly.</h2>
            <div className="mt-5 space-y-2">
              {apiCalls.map((call) => (
                <code key={call} className="block rounded-[1rem] bg-ink px-4 py-3 font-mono text-sm text-paper">{call}</code>
              ))}
            </div>
          </Card>
          <Card>
            <Badge tone="warn">Interpret carefully</Badge>
            <p className="mt-4 font-semibold leading-7 text-ink/78">
              ReliaGuard is for AI evaluation and selective-risk planning. It is not clinical software, not a diagnosis tool, and not causal proof that an intervention will change human behavior. Treat policy outputs as prospective-study design signals.
            </p>
          </Card>
        </div>
      </div>
    </PageShell>
  );
}
