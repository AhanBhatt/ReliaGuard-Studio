"use client";

import { useEffect, useMemo, useState } from "react";
import { Badge, Card, InstrumentCard, PageShell, ValueBar } from "@/components/ui";
import { getJson } from "@/lib/api";
import { decimal, displayDataset, displayLabel, percent } from "@/lib/format";

const presets = [0.01, 0.05, 0.10, 0.20, 0.30];

export default function GatingPage() {
  const [alpha, setAlpha] = useState(0.10);
  const [data, setData] = useState<any[]>([]);
  const [meta, setMeta] = useState<any>({});
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    let active = true;
    setLoading(true);
    getJson(`/conformal-threshold?alpha=${alpha.toFixed(2)}`)
      .then((json) => {
        if (!active) return;
        setData(json.thresholds ?? []);
        setMeta(json);
      })
      .catch(() => {
        if (!active) return;
        setData([]);
        setMeta({});
      })
      .finally(() => active && setLoading(false));
    return () => {
      active = false;
    };
  }, [alpha]);

  const headline = useMemo(() => {
    const rows = data.slice(0, 12);
    if (!rows.length) return {capture: 0, burden: 0, missed: 0};
    return {
      capture: rows.reduce((sum, row) => sum + Number(row.empirical_harmful_capture ?? 0), 0) / rows.length,
      burden: rows.reduce((sum, row) => sum + Number(row.intervention_burden ?? 0), 0) / rows.length,
      missed: rows.reduce((sum, row) => sum + Number(row.missed_harmful_fraction ?? 0), 0) / rows.length
    };
  }, [data]);

  const isPreview = meta.alpha_source === "preview_estimate_from_nearest_artifact";

  return (
    <PageShell eyebrow="Conformal gating dashboard" title="Tune allowed harm and watch the intervention burden move.">
      <div className="grid gap-6 lg:grid-cols-[0.95fr_1.05fr]">
        <InstrumentCard className="instrument-dark text-paper">
          <div className="relative z-10">
            <div className="flex flex-wrap items-center gap-2">
              <Badge tone={isPreview ? "warn" : "safe"}>{isPreview ? "Preview estimate" : "Exact artifact"}</Badge>
              <span className="rounded-full border border-paper/25 bg-paper/15 px-3 py-1 text-xs font-black uppercase tracking-[0.14em] text-bone">
                artifact alpha {meta.served_artifact_alpha ?? "n/a"}
              </span>
            </div>
            <div className="mt-8">
              <div className="flex items-end justify-between gap-4">
                <div>
                  <div className="text-xs font-black uppercase tracking-[0.24em] text-coral">Requested alpha</div>
                  <div className="font-display text-7xl font-black tracking-[-0.07em]">{alpha.toFixed(2)}</div>
                </div>
                <div className="max-w-xs text-sm font-bold leading-6 text-bone">
                  Lower alpha asks the system to miss fewer harmful cases, usually increasing intervention burden.
                </div>
              </div>
              <input
                aria-label="Allowed harmful-case miss rate alpha"
                className="range-control mt-8 w-full"
                type="range"
                min="0.01"
                max="0.30"
                step="0.01"
                value={alpha}
                onChange={(event) => setAlpha(Number(event.target.value))}
              />
              <div className="mt-4 flex flex-wrap gap-2">
                {presets.map((value) => (
                  <button
                    key={value}
                    onClick={() => setAlpha(value)}
                    className={`rounded-full border px-4 py-2 text-xs font-black uppercase tracking-[0.14em] transition ${Math.abs(alpha - value) < 0.001 ? "border-coral bg-coral text-paper" : "border-paper/25 bg-paper/15 text-bone hover:bg-paper/25"}`}
                  >
                    {value.toFixed(2)}
                  </button>
                ))}
              </div>
            </div>
            <div className="mt-8 grid gap-4 sm:grid-cols-3">
              <div className="rounded-[1.5rem] border border-paper/15 bg-bone/12 p-4">
                <div className="text-xs font-black uppercase tracking-[0.18em] text-bone">Harmful capture</div>
                <div className="mt-2 font-display text-4xl font-black text-bone">{percent(headline.capture)}</div>
              </div>
              <div className="rounded-[1.5rem] border border-paper/15 bg-bone/12 p-4">
                <div className="text-xs font-black uppercase tracking-[0.18em] text-bone">Missed harmful</div>
                <div className="mt-2 font-display text-4xl font-black text-coral">{percent(headline.missed)}</div>
              </div>
              <div className="rounded-[1.5rem] border border-paper/15 bg-bone/12 p-4">
                <div className="text-xs font-black uppercase tracking-[0.18em] text-bone">Burden</div>
                <div className="mt-2 font-display text-4xl font-black text-amber">{percent(headline.burden)}</div>
              </div>
            </div>
          </div>
        </InstrumentCard>

        <Card>
          <div className="flex items-start justify-between gap-4">
            <div>
              <h2 className="font-display text-3xl font-black tracking-[-0.04em]">Dataset thresholds</h2>
              <p className="mt-2 max-w-2xl text-sm font-semibold leading-6 text-ink/78">
                {isPreview
                  ? "The current artifact only contains alpha 0.10, so this view uses a monotone preview estimate from the nearest calibrated artifact. Regenerate conformal artifacts for publication-grade reporting."
                  : "These rows come directly from the calibrated conformal artifact."}
              </p>
            </div>
            {loading ? <Badge tone="info">Refreshing</Badge> : <Badge tone={isPreview ? "warn" : "safe"}>{isPreview ? "Exploratory" : "Calibrated"}</Badge>}
          </div>
          <div className="mt-6 space-y-4">
            {data.slice(0, 9).map((row, index) => (
              <div key={`${row.dataset}-${row.target}-${row.model}-${index}`} className="rounded-[1.5rem] border border-ink/10 bg-bone/72 p-4">
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div>
                    <div className="text-xs font-black uppercase tracking-[0.18em] text-coral">{displayDataset(row.dataset)}</div>
                    <div className="mt-1 text-lg font-black">{displayLabel(row.target)} · {displayLabel(row.model)}</div>
                  </div>
                  <div className="rounded-full bg-ink px-3 py-1 text-xs font-black text-paper">threshold {decimal(row.threshold)}</div>
                </div>
                <div className="mt-4 grid gap-4 md:grid-cols-3">
                  <div>
                    <div className="mb-2 flex justify-between text-xs font-black text-ink/75"><span>Capture</span><span>{percent(row.empirical_harmful_capture)}</span></div>
                    <ValueBar value={Number(row.empirical_harmful_capture ?? 0)} tone="safe" />
                  </div>
                  <div>
                    <div className="mb-2 flex justify-between text-xs font-black text-ink/75"><span>Missed</span><span>{percent(row.missed_harmful_fraction)}</span></div>
                    <ValueBar value={Number(row.missed_harmful_fraction ?? 0)} tone="risk" />
                  </div>
                  <div>
                    <div className="mb-2 flex justify-between text-xs font-black text-ink/75"><span>Burden</span><span>{percent(row.intervention_burden)}</span></div>
                    <ValueBar value={Number(row.intervention_burden ?? 0)} tone="info" />
                  </div>
                </div>
              </div>
            ))}
            {!data.length ? (
              <div className="rounded-[1.5rem] border border-coral/25 bg-coral/10 p-5 font-bold text-coral">
                No conformal artifact is available yet. Run <code>nsca run-conformal-risk-control</code> after preparing real data.
              </div>
            ) : null}
          </div>
        </Card>
      </div>
      <Card className="mt-6">
        <h2 className="text-xl font-black">Interpretation boundary</h2>
        <p className="mt-2 font-semibold text-ink/78">
          This dashboard visualizes score-based selective-risk control. It does not prove that an intervention will causally change user behavior. Exact rows should be generated from calibration data at the requested alpha before being used in a paper, report, or deployment review.
        </p>
      </Card>
    </PageShell>
  );
}
