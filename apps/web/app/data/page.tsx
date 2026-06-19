import { Badge, Card, PageShell } from "@/components/ui";
import { getJson } from "@/lib/api";
import { decimal, displayDataset, displayLabel, percent } from "@/lib/format";

export const dynamic = "force-dynamic";

const commands = [
  ["Install", "python -m pip install -e \".[dev]\""],
  ["Download", "nsca download-real-data"],
  ["Prepare", "nsca prepare-real-data"],
  ["Experiments", "nsca run-real-experiments"],
  ["Transfer", "nsca run-cross-dataset"],
  ["Policy", "nsca run-policy-evaluation"],
  ["Conformal", "nsca run-conformal-risk-control"]
];

function formatMetric(key: string, value: unknown) {
  if (value === null || value === undefined || value === "") return "n/a";
  const numeric = Number(value);
  if (Number.isFinite(numeric)) {
    const lower = key.toLowerCase();
    if (lower.includes("rate") || lower.includes("fraction") || lower.includes("capture") || lower.includes("burden")) {
      return percent(numeric, 1);
    }
    if (Math.abs(numeric) >= 1000) return Math.round(numeric).toLocaleString();
    return decimal(numeric, 3);
  }
  return displayLabel(value);
}

export default async function DataPage() {
  let datasets: any[] = [];
  try {
    datasets = (await getJson("/datasets")).datasets ?? [];
  } catch {}
  return (
    <PageShell eyebrow="Data and reproducibility" title="Public evidence enters through canonical schemas, not ad hoc notebooks.">
      <div className="grid gap-6 xl:grid-cols-[1.15fr_0.85fr]">
        <div className="grid gap-4 lg:grid-cols-2">
          {datasets.slice(0, 8).map((dataset, index) => (
            <Card key={index} className="min-w-0">
              <Badge tone="info">Dataset</Badge>
              <h2 className="mt-4 text-3xl font-black tracking-[-0.035em]">{displayDataset(dataset.dataset)}</h2>
              <div className="mt-5 space-y-3">
                {Object.entries(dataset).filter(([key]) => key !== "dataset").slice(0, 6).map(([key, value]) => (
                  <div key={key} className="grid grid-cols-[minmax(0,1fr)_auto] items-start gap-4 rounded-[1.2rem] bg-bone/72 p-3">
                    <div className="min-w-0 text-xs font-black uppercase leading-5 tracking-[0.14em] text-ink/75">{displayLabel(key)}</div>
                    <div className="max-w-[11rem] overflow-hidden text-ellipsis text-right font-mono text-sm font-black text-ink" title={String(value)}>
                      {formatMetric(key, value)}
                    </div>
                  </div>
                ))}
              </div>
            </Card>
          ))}
        </div>
        <Card>
          <Badge tone="safe">Reproducible path</Badge>
          <h2 className="mt-4 text-3xl font-black tracking-[-0.035em]">Run the public pipeline</h2>
          <p className="mt-3 font-semibold leading-7 text-ink/78">Each command writes raw data, prepared tables, models or generated outputs into ignored artifact folders so the public repo stays clean.</p>
          <div className="mt-6 space-y-3">
            {commands.map(([label, command], index) => (
              <div key={command} className="grid grid-cols-[2.2rem_1fr] gap-3 rounded-[1.25rem] bg-ink p-4 text-paper">
                <div className="flex h-8 w-8 items-center justify-center rounded-full bg-coral text-sm font-black">{index + 1}</div>
                <div>
                  <div className="text-xs font-black uppercase tracking-[0.16em] text-paper/90">{label}</div>
                  <code className="font-mono text-sm text-paper">{command}</code>
                </div>
              </div>
            ))}
          </div>
        </Card>
      </div>
    </PageShell>
  );
}
