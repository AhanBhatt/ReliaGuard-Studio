import Link from "next/link";

const links = [
  ["Simulator", "/simulator"],
  ["Gating", "/gating"],
  ["Evaluation", "/evaluation"],
  ["Policy", "/policy"],
  ["Data", "/data"],
  ["Guide", "/guide"],
  ["Limits", "/limitations"]
];

export function Nav() {
  return (
    <header className="sticky top-0 z-50 border-b border-ink/10 bg-bone/80 backdrop-blur-xl">
      <nav className="mx-auto flex max-w-7xl items-center justify-between px-5 py-4 sm:px-6">
        <Link href="/" className="group flex items-center gap-3 text-ink">
          <span className="relative flex h-10 w-10 items-center justify-center rounded-2xl bg-ink text-paper shadow-soft">
            <span className="absolute h-3 w-3 rounded-full bg-coral blur-[2px]" />
            <span className="relative font-display text-sm font-black">RG</span>
          </span>
          <span>
            <span className="block font-display text-xl font-black tracking-[-0.04em]">ReliaGuard Studio</span>
            <span className="hidden text-[0.65rem] font-black uppercase tracking-[0.22em] text-ink/72 sm:block">Human-AI reliance control room</span>
          </span>
        </Link>
        <div className="hidden items-center gap-2 text-sm font-black text-ink/70 md:flex">
          {links.map(([label, href]) => (
            <Link key={href} href={href} className="rounded-full px-3 py-2 transition hover:bg-ink/10 hover:text-ink">
              {label}
            </Link>
          ))}
        </div>
        <div className="flex items-center gap-2 rounded-full border border-moss/20 bg-moss/10 px-3 py-2 text-xs font-black uppercase tracking-[0.15em] text-moss">
          <span className="h-2 w-2 rounded-full bg-moss" />
          Local-first
        </div>
      </nav>
    </header>
  );
}
