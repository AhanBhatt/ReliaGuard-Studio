import { Badge, Card, PageShell, ValueBar } from "@/components/ui";
import { getJson } from "@/lib/api";
import { decimal, displayDataset, displayLabel, percent } from "@/lib/format";

export const dynamic = "force-dynamic";

export default async function PolicyPage() {
  let policies: any[] = [];
  let boundary = "Observational simulation only; not a causal deployment effect.";
  try {
    const json = await getJson("/simulate-policy");
    policies = json.policies ?? [];
    boundary = json.boundary ?? boundary;
  } catch {}

  return (
    <PageShell eyebrow="Policy simulation" title="Compare gating strategies before committing to an intervention design.">
      <div className="grid gap-6 lg:grid-cols-[0.78fr_1.22fr]">
        <Card>
          <Badge tone="warn">Non-causal simulation</Badge>
          <h2 className="mt-4 text-3xl font-black tracking-[-0.035em]">What this page answers</h2>
          <p className="mt-3 leading-7 text-ink/70">
            If a gate intervenes on cases above a risk threshold, how much harmful reliance could be captured, and how much burden would users see? This is a planning tool for experiments, not proof that warnings change behavior.
          </p>
          <div className="mt-6 rounded-[1.5rem] bg-ink p-5 text-paper">
            <div className="text-xs font-black uppercase tracking-[0.2em] text-coral">Boundary</div>
            <p className="mt-3 font-semibold text-paper/85">{boundary}</p>
          </div>
        </Card>

        <div className="grid gap-4 md:grid-cols-2">
          {policies.slice(0, 12).map((policy, index) => (
            <Card key={index}>
              <div className="flex items-start justify-between gap-3">
                <div>
                  <div className="text-xs font-black uppercase tracking-[0.18em] text-coral">{displayDataset(policy.dataset)}</div>
                  <h2 className="mt-2 text-2xl font-black tracking-[-0.025em]">{displayLabel(policy.policy)}</h2>
                </div>
                <span className="rounded-full bg-bone px-3 py-1 text-xs font-black text-ink/75">utility {decimal(policy.expected_utility_proxy ?? policy.utility ?? 0)}</span>
              </div>
              <div className="mt-5 space-y-4">
                <div>
                  <div className="mb-2 flex justify-between text-xs font-black text-ink/75"><span>Final correctness</span><span>{percent(policy.expected_final_correct)}</span></div>
                  <ValueBar value={Number(policy.expected_final_correct ?? 0)} tone="safe" />
                </div>
                <div>
                  <div className="mb-2 flex justify-between text-xs font-black text-ink/75"><span>Overreliance</span><span>{percent(policy.expected_overreliance)}</span></div>
                  <ValueBar value={Number(policy.expected_overreliance ?? 0)} tone="risk" />
                </div>
                <div>
                  <div className="mb-2 flex justify-between text-xs font-black text-ink/75"><span>Intervention burden</span><span>{percent(policy.intervention_burden)}</span></div>
                  <ValueBar value={Number(policy.intervention_burden ?? 0)} tone="info" />
                </div>
              </div>
              {policy.note ? <p className="mt-4 text-xs font-semibold leading-5 text-ink/72">{policy.note}</p> : null}
            </Card>
          ))}
        </div>
      </div>
    </PageShell>
  );
}
