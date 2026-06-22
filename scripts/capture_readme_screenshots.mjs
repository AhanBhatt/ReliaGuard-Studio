import { spawn } from "node:child_process";
import { createRequire } from "node:module";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const root = path.resolve(__dirname, "..");
const webDir = path.join(root, "apps", "web");
const outDir = path.join(root, "docs", "assets", "screenshots");
const requireFromWeb = createRequire(path.join(webDir, "package.json"));
const { chromium } = requireFromWeb("playwright");

const API_PORT = Number(process.env.RELIAGUARD_SCREENSHOT_API_PORT ?? 8000);
const WEB_PORT = Number(process.env.RELIAGUARD_SCREENSHOT_WEB_PORT ?? 3120);
const API_URL = `http://127.0.0.1:${API_PORT}`;
const WEB_URL = `http://127.0.0.1:${WEB_PORT}`;
const spawned = [];

function log(message) {
  console.log(`[screenshots] ${message}`);
}

function startProcess(name, command, args, options) {
  const child = spawn(command, args, {
    cwd: options.cwd,
    env: options.env,
    windowsHide: true,
    stdio: ["ignore", "pipe", "pipe"],
  });
  spawned.push({ name, child });
  log(`${name} pid=${child.pid}`);
  child.stdout.on("data", (chunk) => process.stdout.write(`[${name}] ${chunk}`));
  child.stderr.on("data", (chunk) => process.stderr.write(`[${name}!] ${chunk}`));
  child.on("exit", (code) => {
    if (code !== null && code !== 0) {
      log(`${name} exited with code ${code}`);
    }
  });
  return child;
}

async function waitForHttp(url, label, timeoutMs = 90000) {
  const started = Date.now();
  while (Date.now() - started < timeoutMs) {
    try {
      const response = await fetch(url);
      if (response.ok || response.status < 500) {
        log(`${label} is ready at ${url}`);
        return;
      }
    } catch {
      // Keep waiting.
    }
    await new Promise((resolve) => setTimeout(resolve, 750));
  }
  throw new Error(`${label} did not become ready at ${url}`);
}

async function settle(page) {
  await page.waitForLoadState("domcontentloaded");
  await page.waitForTimeout(1500);
}

async function capture(page, name, url, action) {
  log(`capturing ${name}`);
  await page.goto(url, { waitUntil: "networkidle", timeout: 60000 });
  await page.addStyleTag({
    content: `
      nextjs-portal,
      [data-nextjs-toast],
      [data-nextjs-dev-tools-button],
      [data-nextjs-dev-tools-panel] {
        display: none !important;
      }
    `,
  });
  await settle(page);
  if (action) {
    await action(page);
    await settle(page);
  }
  const target = path.join(outDir, `${name}.png`);
  await page.screenshot({
    path: target,
    fullPage: false,
    animations: "disabled",
    caret: "hide",
  });
  log(`wrote ${path.relative(root, target)}`);
}

function killTree(pid) {
  if (!pid || process.platform !== "win32") {
    return;
  }
  spawn("taskkill", ["/PID", String(pid), "/T", "/F"], {
    windowsHide: true,
    stdio: "ignore",
  });
}

async function main() {
  fs.mkdirSync(outDir, { recursive: true });

  const env = {
    ...process.env,
    PYTHONPATH: `${path.join(root, "src")};${root};${process.env.PYTHONPATH ?? ""}`,
    NEXT_PUBLIC_API_BASE_URL: API_URL,
    PORT: String(WEB_PORT),
    HOSTNAME: "127.0.0.1",
  };

  startProcess("api", "python", [
    "-m",
    "uvicorn",
    "apps.api.main:app",
    "--host",
    "127.0.0.1",
    "--port",
    String(API_PORT),
  ], { cwd: root, env });

  startProcess("web", "cmd.exe", [
    "/d",
    "/s",
    "/c",
    `npm run dev -- --hostname 127.0.0.1 --port ${WEB_PORT}`,
  ], { cwd: webDir, env });

  let browser;
  try {
    await waitForHttp(`${API_URL}/model-card`, "API");
    await waitForHttp(WEB_URL, "web");

    browser = await chromium.launch({ headless: true });
    const page = await browser.newPage({
      viewport: { width: 1440, height: 980 },
      deviceScaleFactor: 1,
    });

    await capture(page, "01-homepage", `${WEB_URL}/`);
    await capture(page, "02-simulator-result", `${WEB_URL}/simulator`, async (p) => {
      await p.getByRole("button", { name: /predict reliance state/i }).click();
      await p.getByText(/Predicted state/i).waitFor({ timeout: 15000 });
    });
    await capture(page, "03-gating-dashboard", `${WEB_URL}/gating`, async (p) => {
      await p.getByRole("button", { name: "0.20" }).click();
      await p.waitForTimeout(1200);
    });
    await capture(page, "04-evaluation-lab", `${WEB_URL}/evaluation`);
    await capture(page, "05-upload-audit", `${WEB_URL}/upload`, async (p) => {
      await p.getByRole("button", { name: /generate audit report/i }).click();
      const report = p.getByText(/scored interactions/i);
      await report.waitFor({ timeout: 15000 });
      await report.scrollIntoViewIfNeeded();
    });
    await capture(page, "06-review-queue", `${WEB_URL}/review`, async (p) => {
      await p.getByText(/cases waiting or labelled/i).waitFor({ timeout: 15000 });
    });
    await capture(page, "07-monitoring", `${WEB_URL}/monitoring`, async (p) => {
      await p.getByText(/logged events/i).waitFor({ timeout: 15000 });
    });
  } finally {
    if (browser) {
      await browser.close();
    }
    for (const { child } of spawned.reverse()) {
      if (!child.killed) {
        killTree(child.pid);
      }
    }
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
