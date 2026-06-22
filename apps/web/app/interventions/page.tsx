import { Badge, Card, PageShell } from "@/components/ui";
import { getJson } from "@/lib/api";
import { displayLabel } from "@/lib/format";

export const dynamic = "force-dynamic";

export default async function InterventionsPage() {
  let interventions: any[] = [];
  try {
    interventions = (await getJson("/v1/interventions")).interventions ?? [];
  } catch {}

  return (
    <PageShell eyebrow="Intervention library" title="Turn reliance-risk patterns into concrete product copy and routing actions.">
      <div className="grid gap-6 lg:grid-cols-[0.8fr_1.2fr]">
        <Card>
          <Badge tone="safe">Editable templates</Badge>
          <h2 className="mt-4 text-3xl font-black tracking-[-0.035em]">From detection to intervention design</h2>
          <p className="mt-3 font-semibold leading-7 text-ink/74">
            ReliaGuard does not only say “risk is high.” It maps risk patterns to verification prompts, uncertainty cues, delay flows, or review routing. Product teams can tune this language before moving from shadow mode to guardrail mode.
          </p>
          <div className="mt-6 rounded-[1.5rem] bg-ink p-5 text-paper">
            <div className="text-xs font-black uppercase tracking-[0.18em] text-coral">Safety boundary</div>
            <p className="mt-3 font-semibold leading-6 text-paper/86">Templates are intervention candidates. They require prospective validation before claiming causal behavior change.</p>
          </div>
        </Card>
        <div className="grid gap-4">
          {interventions.map((item) => (
            <Card key={item.key}>
              <div className="grid gap-5 md:grid-cols-[0.75fr_1.25fr_0.5fr]">
                <div>
                  <div className="text-xs font-black uppercase tracking-[0.16em] text-coral">Risk pattern</div>
                  <h2 className="mt-2 text-2xl font-black tracking-[-0.025em]">{item.risk_pattern}</h2>
                </div>
                <div className="rounded-[1.25rem] bg-bone/72 p-4">
                  <div className="text-xs font-black uppercase tracking-[0.16em] text-ink/62">User-facing intervention</div>
                  <p className="mt-2 font-semibold leading-7 text-ink/82">“{item.intervention}”</p>
                </div>
                <div className="flex items-center justify-start md:justify-end">
                  <Badge tone={item.action === "request_verification" || item.action === "route_to_review" ? "risk" : "info"}>{displayLabel(item.action)}</Badge>
                </div>
              </div>
            </Card>
          ))}
        </div>
      </div>
    </PageShell>
  );
}
