import clsx from "clsx";
import Link from "next/link";
import type { ReactNode } from "react";

export function Card({ children, className }: { children: ReactNode; className?: string }) {
  return <section className={clsx("glass rounded-[2rem] p-6 shadow-soft", className)}>{children}</section>;
}

export function InstrumentCard({ children, className }: { children: ReactNode; className?: string }) {
  return <section className={clsx("glass instrument rounded-[2.25rem] p-7 shadow-instrument", className)}>{children}</section>;
}

export function ButtonLink({ href, children }: { href: string; children: ReactNode }) {
  return (
    <Link href={href} className="inline-flex rounded-full bg-ink px-5 py-3 text-sm font-black uppercase tracking-[0.12em] text-paper shadow-soft transition hover:-translate-y-0.5 hover:bg-moss">
      {children}
    </Link>
  );
}

export function Metric({ label, value, note }: { label: string; value: string; note?: string }) {
  return (
    <div className="rounded-[1.35rem] border border-ink/10 bg-bone/65 p-4 shadow-[inset_0_1px_0_rgba(255,255,255,0.6)]">
      <div className="font-display text-3xl font-black text-ink">{value}</div>
      <div className="mt-1 text-xs font-black uppercase tracking-[0.12em] text-ink/75">{label}</div>
      {note ? <div className="mt-2 text-xs font-bold text-ink/72">{note}</div> : null}
    </div>
  );
}

export function Badge({ children, tone = "neutral" }: { children: ReactNode; tone?: "neutral" | "risk" | "safe" | "warn" | "info" }) {
  const styles = {
    neutral: "border-ink/20 bg-bone/80 text-ink",
    risk: "border-coral/35 bg-coral/15 text-coral",
    safe: "border-moss/35 bg-moss/15 text-moss",
    warn: "border-amber/50 bg-amber/85 text-night",
    info: "border-sky/35 bg-sky/15 text-sky"
  };
  return <span className={clsx("inline-flex rounded-full border px-3 py-1 text-xs font-black uppercase tracking-[0.14em]", styles[tone])}>{children}</span>;
}

export function ValueBar({ value, tone = "risk" }: { value: number; tone?: "risk" | "safe" | "info" }) {
  const color = tone === "safe" ? "bg-moss" : tone === "info" ? "bg-sky" : "bg-coral";
  return (
    <div className="h-2 overflow-hidden rounded-full bg-ink/10">
      <div className={clsx("h-full rounded-full", color)} style={{width: `${Math.max(0, Math.min(1, value)) * 100}%`}} />
    </div>
  );
}

export function PageShell({ eyebrow, title, children }: { eyebrow: string; title: string; children: ReactNode }) {
  return (
    <main className="mx-auto max-w-7xl px-5 py-10 sm:px-6 lg:py-12">
      <p className="text-xs font-black uppercase tracking-[0.34em] text-coral">{eyebrow}</p>
      <h1 className="mt-3 max-w-5xl font-display text-5xl font-black leading-[0.96] tracking-[-0.045em] text-ink md:text-7xl">{title}</h1>
      <div className="mt-8">{children}</div>
    </main>
  );
}
