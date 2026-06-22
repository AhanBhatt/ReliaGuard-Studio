import Link from "next/link";

const primaryLinks = [
  ["Audit", "/upload"],
  ["Projects", "/projects"],
  ["Simulator", "/simulator"],
  ["Gating", "/gating"],
  ["Review", "/review"],
  ["Monitor", "/monitoring"]
];

const secondaryLinks = [
  ["Replay", "/replay"],
  ["Interventions", "/interventions"],
  ["Evaluation", "/evaluation"],
  ["Policy", "/policy"],
  ["Data", "/data"],
  ["Guide", "/guide"],
  ["Limits", "/limitations"]
];

export function Nav() {
  return (
    <header className="sticky top-0 z-50 border-b border-ink/10 bg-bone/80 backdrop-blur-xl">
      <nav className="mx-auto flex max-w-7xl items-center gap-5 px-5 py-3 sm:px-6">
        <Link href="/" className="group flex min-w-[15rem] items-center gap-3 text-ink">
          <span className="relative flex h-11 w-11 shrink-0 items-center justify-center rounded-2xl bg-ink text-paper shadow-soft">
            <span className="absolute h-3 w-3 rounded-full bg-coral blur-[2px]" />
            <span className="relative font-display text-sm font-black">RG</span>
          </span>
          <span className="leading-tight">
            <span className="block whitespace-nowrap font-display text-xl font-black tracking-[-0.04em]">ReliaGuard Studio</span>
            <span className="hidden whitespace-nowrap text-[0.63rem] font-black uppercase tracking-[0.18em] text-ink/68 sm:block">Reliance observability platform</span>
          </span>
        </Link>
        <div className="hidden flex-1 items-center justify-center gap-1 text-sm font-black text-ink/70 lg:flex">
          {primaryLinks.map(([label, href]) => (
            <Link key={href} href={href} className="rounded-full px-3.5 py-2 transition hover:bg-ink/10 hover:text-ink">
              {label}
            </Link>
          ))}
          <details className="group relative">
            <summary className="list-none rounded-full px-3.5 py-2 transition hover:bg-ink/10 hover:text-ink marker:content-none">
              More
            </summary>
            <div className="absolute right-0 top-11 z-50 grid min-w-[14rem] gap-1 rounded-[1.35rem] border border-ink/10 bg-bone/95 p-2 shadow-soft backdrop-blur-xl">
              {secondaryLinks.map(([label, href]) => (
                <Link key={href} href={href} className="rounded-[1rem] px-3 py-2 text-sm font-black text-ink/76 transition hover:bg-ink hover:text-paper">
                  {label}
                </Link>
              ))}
            </div>
          </details>
        </div>
        <div className="ml-auto flex shrink-0 items-center gap-2 rounded-full border border-moss/20 bg-moss/10 px-4 py-2 text-xs font-black uppercase tracking-[0.14em] text-moss">
          <span className="h-2 w-2 rounded-full bg-moss" />
          Local-first
        </div>
      </nav>
    </header>
  );
}
