import { Badge, Card, PageShell, ValueBar } from "@/components/ui";
import { getJson } from "@/lib/api";
import { decimal, displayDataset, displayLabel } from "@/lib/format";

export const dynamic = "force-dynamic";

export default async function EvaluationPage() {
  let lab: any = {model_results: [], calibration: [], cross_dataset_transfer: []};
  try {
    lab = await getJson("/evaluation-lab");
  } catch {}
  const rows = lab.model_results?.slice(0, 10) ?? [];
  const calibration = lab.calibration?.slice(0, 4) ?? [];

  return (
    <PageShell eyebrow="Model evaluation lab" title="Model performance is useful only when calibration and split rigor are visible.">
      <div className="grid gap-6 xl:grid-cols-[1.2fr_0.8fr]">
        <Card>
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div>
              <Badge tone="info">Strict comparison</Badge>
              <h2 className="mt-4 text-3xl font-black tracking-[-0.035em]">Predictive metrics</h2>
              <p className="mt-2 text-sm font-semibold text-ink/78">Baselines are shown alongside ReliaGuard variants; the product value is calibrated diagnosis, not hiding baseline wins.</p>
            </div>
          </div>
          <div className="mt-6 space-y-3">
            {rows.map((row: any, index: number) => (
              <div key={index} className="rounded-[1.35rem] border border-ink/10 bg-bone/72 p-4">
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div>
                    <div className="text-xs font-black uppercase tracking-[0.18em] text-coral">{displayDataset(row.dataset)} · {displayLabel(row.split)}</div>
                    <div className="mt-1 text-lg font-black">{displayLabel(row.model)}</div>
                    <div className="text-sm font-semibold text-ink/72">{displayLabel(row.target)}</div>
                  </div>
                  <div className="grid grid-cols-2 gap-2 text-right text-xs font-black text-ink/78">
                    <span>AUROC <b className="text-ink">{decimal(row.auroc)}</b></span>
                    <span>AUPRC <b className="text-ink">{decimal(row.auprc)}</b></span>
                    <span>Brier <b className="text-ink">{decimal(row.brier_score)}</b></span>
                    <span>ECE <b className="text-ink">{decimal(row.ece)}</b></span>
                  </div>
                </div>
                <div className="mt-4">
                  <ValueBar value={Number(row.auroc ?? 0)} tone="info" />
                </div>
              </div>
            ))}
          </div>
        </Card>

        <Card>
          <Badge tone="warn">Calibration lens</Badge>
          <h2 className="mt-4 text-3xl font-black tracking-[-0.035em]">Reliability checks</h2>
          <p className="mt-2 text-sm font-semibold leading-6 text-ink/78">Lower Brier and ECE values are better. Mixed calibration is reported as a product boundary, not brushed away.</p>
          <div className="mt-6 space-y-4">
            {calibration.map((row: any, index: number) => (
              <div key={index} className="rounded-[1.35rem] bg-ink p-4 text-paper">
                <div className="font-black">{displayDataset(row.dataset)} · {displayLabel(row.model)}</div>
                <div className="mt-3 grid grid-cols-2 gap-3 text-sm font-semibold text-paper/85">
                  <div>ECE <b className="text-paper">{decimal(row.ece)}</b></div>
                  <div>Brier <b className="text-paper">{decimal(row.brier_score)}</b></div>
                  <div>Slope <b className="text-paper">{decimal(row.calibration_slope)}</b></div>
                  <div>Intercept <b className="text-paper">{decimal(row.calibration_intercept)}</b></div>
                </div>
              </div>
            ))}
          </div>
        </Card>
      </div>
    </PageShell>
  );
}
