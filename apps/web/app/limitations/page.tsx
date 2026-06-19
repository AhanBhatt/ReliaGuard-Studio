import { Card, PageShell } from "@/components/ui";

const limits = [
  "Not a clinical, diagnostic, or cognitive-decline tool.",
  "No completed prospective randomized intervention trial is reported.",
  "Conformal gating is exchangeability-bound and non-causal.",
  "Learning/process datasets do not support delayed recall, transfer, or long-term learning claims.",
  "Cross-dataset transfer is partial and reported as a boundary, not hidden."
];

export default function LimitationsPage() {
  return (
    <PageShell eyebrow="Safety and limitations" title="ReliaGuard Studio is honest about what it can and cannot claim.">
      <Card>
        <ul className="space-y-4">
          {limits.map((limit) => (
            <li key={limit} className="rounded-2xl bg-white/65 p-4 text-lg font-bold text-ink/75">{limit}</li>
          ))}
        </ul>
      </Card>
    </PageShell>
  );
}
