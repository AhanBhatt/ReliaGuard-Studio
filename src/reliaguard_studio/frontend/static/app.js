const taskFamily = document.getElementById("taskFamily");
const conditionId = document.getElementById("conditionId");
const loadSample = document.getElementById("loadSample");
const scoreSession = document.getElementById("scoreSession");
const exportReport = document.getElementById("exportReport");
const taskCard = document.getElementById("taskCard");
const sessionCard = document.getElementById("sessionCard");
const rulesList = document.getElementById("rulesList");
const counterfactuals = document.getElementById("counterfactuals");
const metricGlossary = document.getElementById("metricGlossary");
const benchmarkSnapshot = document.getElementById("benchmarkSnapshot");
const conditionSummary = document.getElementById("conditionSummary");

const API_BASE = window.location.protocol === "file:" ? "http://127.0.0.1:8000" : "";

let currentSession = null;
let currentTask = null;
let currentReport = null;

async function fetchJSON(url, options = {}) {
  const response = await fetch(`${API_BASE}${url}`, options);
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }
  return response.json();
}

function populateSelect(select, values, labelKey = "name", valueKey = "id") {
  select.innerHTML = `<option value="">Any</option>`;
  values.forEach((item) => {
    const option = document.createElement("option");
    option.value = item[valueKey];
    option.textContent = item[labelKey];
    select.appendChild(option);
  });
}

function renderTask(task) {
  taskCard.classList.remove("empty");
  taskCard.textContent = [
    `Task ID: ${task.task_id}`,
    `Family: ${task.family}`,
    `Concept: ${task.concept}`,
    ``,
    `Prompt: ${task.prompt}`,
    ``,
    `Reference answer: ${task.reference_answer}`,
    ``,
    `Flawed AI stressor: ${task.flawed_ai_answer}`,
    ``,
    `Stress-test transfer probe: ${task.transfer_prompt}`,
    ``,
    `Stress-test recall probe: ${task.recall_prompt}`,
  ].join("\n");
}

function renderSession(session) {
  sessionCard.classList.remove("empty");
  sessionCard.textContent = [
    `Session: ${session.session_id}`,
    `Condition: ${session.condition_id}`,
    `Task family: ${session.task_family}`,
    ``,
    `Immediate success probability: ${session.immediate_success_probability.toFixed(3)}`,
    `Delayed recall score: ${session.delayed_recall_score.toFixed(3)}`,
    `Transfer score: ${session.transfer_score.toFixed(3)}`,
    `Verification robustness: ${session.verification_robustness.toFixed(3)}`,
    `Cognitive offloading index: ${session.cognitive_offloading_index.toFixed(3)}`,
    `Copy-paste dependence: ${session.copy_paste_dependence.toFixed(3)}`,
    `Source checking rate: ${session.source_checking_rate.toFixed(3)}`,
    `Calibration error: ${session.calibration_error.toFixed(3)}`,
    `Latent stress-test driver: ${session.latent_driver}`,
  ].join("\n");
}

function renderReport(report) {
  currentReport = report;
  document.getElementById("neuralScore").textContent = report.neural_overreliance_probability.toFixed(3);
  document.getElementById("symbolicScore").textContent = report.symbolic_overreliance_probability.toFixed(3);
  document.getElementById("fusionScore").textContent = report.fusion_overreliance_probability.toFixed(3);
  document.getElementById("uncertaintyScore").textContent = report.uncertainty.toFixed(3);

  rulesList.classList.remove("empty");
  rulesList.innerHTML = report.symbolic.explanation.top_rules.map((rule) => `
    <div class="rule-item">
      <div class="badge">${rule.group}</div>
      <strong>${rule.rule_name}</strong>
      <div>Activation: ${rule.activation.toFixed(3)} | Weight: ${rule.signed_weight.toFixed(3)} | Confidence: ${rule.confidence.toFixed(3)}</div>
      <small>${rule.description}</small>
    </div>
  `).join("");

  counterfactuals.classList.remove("empty");
  counterfactuals.innerHTML = [
    `<p>${report.synthetic_validation_notice}</p>`,
    `<ul>${report.symbolic.explanation.counterfactuals.map((item) => `<li>${item}</li>`).join("")}</ul>`,
  ].join("");
}

function renderDashboard(data) {
  const summary = data.summary;
  benchmarkSnapshot.classList.remove("empty");
  benchmarkSnapshot.innerHTML = [
    `<div><strong>Stress-test participants:</strong> ${summary.n_users}</div>`,
    `<div><strong>Sessions per participant:</strong> ${summary.sessions_per_user}</div>`,
    `<div><strong>Total sessions:</strong> ${summary.total_sessions}</div>`,
    `<div><strong>Best macro model:</strong> ${summary.best_classification_model.model}</div>`,
    `<div><strong>Macro AUROC:</strong> ${summary.best_classification_model.auroc.toFixed(3)}</div>`,
    `<div><strong>Explanation faithfulness:</strong> ${summary.explanation_faithfulness.toFixed(3)}</div>`,
  ].join("");

  conditionSummary.innerHTML = data.condition_effects.map((row) => `
    <tr>
      <td>${row.condition_id}</td>
      <td>${Number(row.immediate_success).toFixed(3)}</td>
      <td>${Number(row.delayed_recall_success).toFixed(3)}</td>
      <td>${Number(row.transfer_success).toFixed(3)}</td>
      <td>${Number(row.verification_failure).toFixed(3)}</td>
      <td>${Number(row.overreliance_risk).toFixed(3)}</td>
    </tr>
  `).join("");
}

async function initialize() {
  const config = await fetchJSON("/api/config");
  const dashboard = await fetchJSON("/api/dashboard/summary");
  populateSelect(taskFamily, config.task_families);
  populateSelect(conditionId, config.assistance_conditions);
  metricGlossary.innerHTML = config.metrics
    .slice(0, 10)
    .map((metric) => `<li><strong>${metric.name}:</strong> ${metric.description}</li>`)
    .join("");
  renderDashboard(dashboard);
}

loadSample.addEventListener("click", async () => {
  const params = new URLSearchParams();
  if (taskFamily.value) params.set("task_family", taskFamily.value);
  if (conditionId.value) params.set("condition_id", conditionId.value);
  const payload = await fetchJSON(`/api/demo/sample-session?${params.toString()}`);
  currentSession = payload.session;
  currentTask = payload.task;
  renderTask(currentTask);
  renderSession(currentSession);
});

scoreSession.addEventListener("click", async () => {
  if (!currentSession) return;
  const report = await fetchJSON("/api/score", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({session_id: currentSession.session_id}),
  });
  renderReport(report);
});

exportReport.addEventListener("click", () => {
  if (!currentReport) return;
  const blob = new Blob([JSON.stringify(currentReport, null, 2)], {type: "application/json"});
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `${currentSession?.session_id || "air_bench_session"}_report.json`;
  anchor.click();
  URL.revokeObjectURL(url);
});

initialize().catch((error) => {
  console.error(error);
  taskCard.textContent = "Failed to load config. Start the FastAPI server with `nsca serve-api` and reload the page.";
  sessionCard.textContent = error.message;
  benchmarkSnapshot.textContent = "Stress-test summary unavailable until the local API is running.";
});
