import { Badge, Card, PageShell } from "@/components/ui";

const quickstart = [
  ["1. Create a project", "Open Projects and choose Audit, Shadow, or Guardrail mode for your AI product."],
  ["2. Upload logs", "Go to Upload Logs, load CSV/JSON/JSONL, map columns, and generate an audit report."],
  ["3. Review flagged cases", "Use Review Queue to label true harmful reliance, false positives, bad ground truth, and burden."],
  ["4. Replay thresholds", "Run Replay Logs to ask what would have happened under a different alpha or policy."],
  ["5. Stream live events", "Use the Python or TypeScript SDK to log interactions during shadow mode."],
  ["6. Call guardrails", "In guardrail mode, call /v1/guardrail/check before finalization and apply the recommended action."]
];

const schema = [
  ["user_id", "agent_id, learner_id, reviewer_id"],
  ["task_id", "ticket_id, question_id, case_id"],
  ["initial_answer", "human_first_decision"],
  ["initial_confidence", "confidence_before_ai"],
  ["ai_advice", "model_recommendation"],
  ["ai_confidence", "model_confidence"],
  ["final_answer", "human_final_decision"],
  ["ground_truth", "true_label, QA outcome, test result"],
  ["context", "domain, task type, model name"]
];

const apiCalls = [
  "POST /v1/guardrail/check",
  "POST /v1/events/log",
  "POST /v1/ingest/validate",
  "GET /v1/review-queue",
  "POST /v1/replay",
  "GET /v1/monitoring",
  "GET /v1/interventions"
];

export default function GuidePage() {
  return (
    <PageShell eyebrow="User guide" title="Audit logs, stream events, and deploy reliance guardrails in stages.">
      <div className="grid gap-6 lg:grid-cols-[1.05fr_0.95fr]">
        <Card>
          <Badge tone="safe">5-minute audit</Badge>
          <div className="mt-6 space-y-4">
            {quickstart.map(([title, body]) => (
              <div key={title} className="grid grid-cols-[2.6rem_1fr] gap-4 rounded-[1.5rem] bg-bone/72 p-4">
                <div className="flex h-10 w-10 items-center justify-center rounded-full bg-coral font-black text-paper">{title.split(".")[0]}</div>
                <div>
                  <h2 className="text-xl font-black">{title.replace(/^\d+\.\s*/, "")}</h2>
                  <p className="mt-1 font-semibold leading-6 text-ink/78">{body}</p>
                </div>
              </div>
            ))}
          </div>
        </Card>
        <div className="grid gap-6">
          <Card>
            <Badge tone="info">Canonical schema</Badge>
            <div className="mt-5 space-y-2">
              {schema.map(([field, examples]) => (
                <div key={field} className="grid grid-cols-[11rem_1fr] gap-3 rounded-[1rem] bg-bone/72 p-3">
                  <code className="font-mono text-sm font-black text-ink">{field}</code>
                  <span className="text-sm font-semibold text-ink/72">{examples}</span>
                </div>
              ))}
            </div>
          </Card>
          <Card>
            <Badge tone="risk">Guardrail API</Badge>
            <pre className="mt-5 overflow-auto rounded-[1.5rem] bg-ink p-5 text-sm font-semibold leading-6 text-paper">{`POST /v1/guardrail/check
{
  "project_id": "support-copilot",
  "user_id": "agent_123",
  "task_id": "ticket_456",
  "initial_answer": "refund",
  "initial_confidence": 0.72,
  "ai_advice": "deny_refund",
  "ai_confidence": 0.88,
  "final_answer": "deny_refund",
  "ground_truth": "refund",
  "context": {"domain": "customer_support"}
}`}</pre>
          </Card>
        </div>
      </div>

      <div className="mt-6 grid gap-6 lg:grid-cols-2">
        <Card>
          <Badge tone="warn">SDK snippets</Badge>
          <pre className="mt-5 overflow-auto rounded-[1.5rem] bg-ink p-5 text-sm font-semibold leading-6 text-paper">{`from reliaguard import ReliaGuard

rg = ReliaGuard(api_key="local-dev")
rg.log_interaction(
    project_id="support-copilot",
    user_id="agent_123",
    task_id="ticket_456",
    initial_answer="refund",
    initial_confidence=0.72,
    ai_advice="deny_refund",
    ai_confidence=0.88,
    final_answer="deny_refund",
    ground_truth="refund",
    context={"domain": "customer_support"}
)`}</pre>
        </Card>
        <Card>
          <Badge tone="info">TypeScript</Badge>
          <pre className="mt-5 overflow-auto rounded-[1.5rem] bg-ink p-5 text-sm font-semibold leading-6 text-paper">{`await reliaguard.logInteraction({
  projectId: "ai-tutor",
  userId: "learner_42",
  taskId: "algebra_17",
  initialAnswer: "x = 4",
  initialConfidence: 0.61,
  aiAdvice: "x = 5",
  aiConfidence: 0.84,
  finalAnswer: "x = 5",
  groundTruth: "x = 4",
  context: { domain: "tutoring" }
});`}</pre>
        </Card>
      </div>

      <div className="mt-6 grid gap-6 lg:grid-cols-[0.8fr_1.2fr]">
        <Card>
          <Badge tone="safe">API map</Badge>
          <div className="mt-5 space-y-2">
            {apiCalls.map((call) => <code key={call} className="block rounded-[1rem] bg-bone/72 px-4 py-3 font-mono text-sm font-black text-ink">{call}</code>)}
          </div>
        </Card>
        <Card>
          <Badge tone="warn">Interpretation limits</Badge>
          <h2 className="mt-4 text-3xl font-black tracking-[-0.035em]">What ReliaGuard does not prove</h2>
          <div className="mt-4 space-y-3 font-semibold leading-7 text-ink/76">
            <p>ReliaGuard is an observability and selective-risk platform, not clinical software and not a diagnosis tool.</p>
            <p>Historical replay estimates which cases would have been flagged. It does not prove that showing an intervention would causally change behavior.</p>
            <p>Lower alpha asks the gate to miss fewer harmful cases, but usually increases intervention burden. Teams should start in audit or shadow mode before enabling guardrail actions.</p>
          </div>
        </Card>
      </div>
    </PageShell>
  );
}
