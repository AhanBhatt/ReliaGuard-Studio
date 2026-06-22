"use client";

import { useEffect, useState } from "react";
import { Badge, ButtonLink, Card, PageShell } from "@/components/ui";
import { getJson, postJson } from "@/lib/api";
import { displayLabel } from "@/lib/format";

export default function ProjectsPage() {
  const [projects, setProjects] = useState<any[]>([]);
  const [name, setName] = useState("Customer Support Copilot");
  const [description, setDescription] = useState("Audit refund, escalation and QA decisions before enabling guardrails.");
  const [mode, setMode] = useState("audit");

  async function refresh() {
    const json = await getJson("/v1/projects");
    setProjects(json.projects ?? []);
  }

  useEffect(() => {
    refresh().catch(() => {});
  }, []);

  async function create() {
    await postJson("/v1/projects", {name, description, mode});
    await refresh();
  }

  return (
    <PageShell eyebrow="Project workspaces" title="Organize audits, thresholds, policies and review feedback by product.">
      <div className="grid gap-6 xl:grid-cols-[0.8fr_1.2fr]">
        <Card>
          <Badge tone="safe">New workspace</Badge>
          <h2 className="mt-4 text-3xl font-black tracking-[-0.035em]">Create a project</h2>
          <div className="mt-6 grid gap-4">
            <label className="grid gap-2 text-sm font-black uppercase tracking-[0.14em] text-ink/70">
              Name
              <input value={name} onChange={(event) => setName(event.target.value)} className="rounded-[1rem] border border-ink/12 bg-bone/85 px-4 py-3 normal-case tracking-normal text-ink" />
            </label>
            <label className="grid gap-2 text-sm font-black uppercase tracking-[0.14em] text-ink/70">
              Description
              <textarea value={description} onChange={(event) => setDescription(event.target.value)} className="min-h-28 rounded-[1rem] border border-ink/12 bg-bone/85 px-4 py-3 normal-case tracking-normal text-ink" />
            </label>
            <label className="grid gap-2 text-sm font-black uppercase tracking-[0.14em] text-ink/70">
              Deployment mode
              <select value={mode} onChange={(event) => setMode(event.target.value)} className="rounded-[1rem] border border-ink/12 bg-bone/85 px-4 py-3 normal-case tracking-normal text-ink">
                <option value="audit">Audit Mode - upload historical logs</option>
                <option value="shadow">Shadow Mode - score live events without intervening</option>
                <option value="guardrail">Guardrail Mode - return intervention actions</option>
              </select>
            </label>
            <button onClick={create} className="rounded-full bg-ink px-5 py-3 text-sm font-black uppercase tracking-[0.14em] text-paper shadow-soft transition hover:bg-coral">
              Create workspace
            </button>
          </div>
        </Card>

        <div className="grid gap-4 lg:grid-cols-2">
          {projects.map((project) => (
            <Card key={project.project_id}>
              <div className="flex flex-wrap items-start justify-between gap-3">
                <Badge tone={project.mode === "guardrail" ? "risk" : project.mode === "shadow" ? "warn" : "info"}>{displayLabel(project.mode)} mode</Badge>
                <span className="rounded-full bg-bone px-3 py-1 font-mono text-xs font-black text-ink/70">{project.project_id}</span>
              </div>
              <h2 className="mt-4 text-3xl font-black tracking-[-0.035em]">{project.name}</h2>
              <p className="mt-3 min-h-16 font-semibold leading-7 text-ink/72">{project.description}</p>
              <div className="mt-5 grid grid-cols-2 gap-3 text-sm font-bold text-ink/74">
                <div className="rounded-[1rem] bg-bone/72 p-3">Thresholds<br /><b>{Object.keys(project.saved_thresholds ?? {}).length}</b></div>
                <div className="rounded-[1rem] bg-bone/72 p-3">Policies<br /><b>{(project.saved_policies ?? []).length}</b></div>
                <div className="rounded-[1rem] bg-bone/72 p-3">Model versions<br /><b>{(project.model_versions ?? []).length}</b></div>
                <div className="rounded-[1rem] bg-bone/72 p-3">Reports<br /><b>{(project.audit_reports ?? []).length}</b></div>
              </div>
              <div className="mt-5 flex flex-wrap gap-3">
                <ButtonLink href="/upload">Upload logs</ButtonLink>
                <ButtonLink href="/monitoring">Monitor</ButtonLink>
              </div>
            </Card>
          ))}
        </div>
      </div>
    </PageShell>
  );
}
