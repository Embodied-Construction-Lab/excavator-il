"use strict";

const UI_HEADER = {"X-Excavator-UI": "1"};
const terminalStages = new Set(["idle", "completed", "failed", "cancelled"]);
const stageLabels = {
  idle: "空闲", starting: "启动中", preflight: "环境检查",
  rl_positioning: "RL 定位", collector_starting: "Collector 启动",
  manual_positioning: "手工预定位", recorder_standby: "等待 deadman",
  recording: "记录中", review: "结果确认", finalizing: "保存中",
  validating: "校验中", completed: "已完成", stopping: "停止中",
  failed: "失败", cancelled: "已取消"
};
const progressOrder = ["position", "standby", "record", "review", "done"];
const stageProgress = {
  rl_positioning: 0, collector_starting: 0, manual_positioning: 0,
  recorder_standby: 1, recording: 2, review: 3,
  finalizing: 4, validating: 4, completed: 4
};

const state = {selectedMode: "rl", config: null, snapshot: null};
const $ = (id) => document.getElementById(id);

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.detail || `HTTP ${response.status}`);
  return payload;
}

async function command(path, body) {
  const headers = {...UI_HEADER};
  const options = {method: "POST", headers};
  if (body !== undefined) {
    headers["Content-Type"] = "application/json";
    options.body = JSON.stringify(body);
  }
  try {
    renderSnapshot(await api(path, options));
  } catch (error) {
    toast(error.message, true);
  }
}

function renderConfig(config) {
  state.config = config;
  $("operator-id").textContent = config.operator_id;
  $("task-name").textContent = config.task;
  $("dig-target").textContent = config.dig_target_m.map(value => Number(value).toFixed(2)).join(", ");
  $("orin-host").textContent = config.orin_host;
  const image = $("camera-preview");
  image.src = config.camera_preview_url;
  image.addEventListener("load", () => {
    image.classList.add("ready");
    $("camera-placeholder").classList.add("hidden");
    $("camera-state").textContent = "实时";
  });
  image.addEventListener("error", () => {
    image.classList.remove("ready");
    $("camera-placeholder").classList.remove("hidden");
    $("camera-state").textContent = "等待 Collector";
  });
  if (config.visualization_url) {
    const link = $("visualization-link");
    link.href = config.visualization_url;
    link.classList.remove("hidden");
    $("visualization-note").classList.add("hidden");
  }
}

function renderSnapshot(snapshot) {
  state.snapshot = snapshot;
  const stage = snapshot.stage || "idle";
  const active = !terminalStages.has(stage);
  $("stage-label").textContent = stageLabels[stage] || stage;
  $("status-dot").className = `status-dot${active ? " active" : ""}${stage === "failed" ? " error" : ""}`;
  $("start-button").disabled = active;
  $("stop-button").disabled = !active || stage === "stopping";
  document.querySelectorAll(".mode-card").forEach(card => { card.disabled = active; });

  const manual = stage === "manual_positioning";
  const review = stage === "review";
  $("stage-actions").classList.toggle("hidden", !manual && !review);
  $("manual-actions").classList.toggle("hidden", !manual);
  $("review-actions").classList.toggle("hidden", !review);

  const logs = Array.isArray(snapshot.logs) ? snapshot.logs : [];
  $("log-output").textContent = logs.length ? logs.join("\n") : "等待采集任务…";
  $("log-output").scrollTop = $("log-output").scrollHeight;
  $("episode-path").textContent = snapshot.episode_path || "";
  $("episode-path").title = snapshot.episode_path || "";
  $("error-banner").textContent = snapshot.error || "";
  $("error-banner").classList.toggle("hidden", !snapshot.error);
  renderProgress(stage);
}

function renderProgress(stage) {
  const current = stageProgress[stage] ?? -1;
  document.querySelectorAll("[data-stage-group]").forEach((node, index) => {
    node.classList.toggle("active", index === current);
    node.classList.toggle("done", index < current || stage === "completed");
  });
  document.querySelectorAll(".timeline i").forEach((node, index) => {
    node.classList.toggle("done", index < current || stage === "completed");
  });
}

function toast(message, isError = false) {
  const node = $("toast");
  node.textContent = message;
  node.style.background = isError ? "#8d2e2b" : "#18362d";
  node.classList.add("visible");
  window.setTimeout(() => node.classList.remove("visible"), 3200);
}

function bindActions() {
  document.querySelectorAll(".mode-card").forEach(card => card.addEventListener("click", () => {
    state.selectedMode = card.dataset.mode;
    document.querySelectorAll(".mode-card").forEach(node => node.classList.toggle("selected", node === card));
  }));
  $("start-button").addEventListener("click", () => command("/api/collection/start", {positioning_mode: state.selectedMode}));
  $("stop-button").addEventListener("click", () => command("/api/collection/stop"));
  $("manual-complete-button").addEventListener("click", () => command("/api/collection/manual-complete"));
  document.querySelectorAll("[data-outcome]").forEach(button => button.addEventListener("click", () => {
    command("/api/collection/outcome", {outcome: button.dataset.outcome});
  }));
}

async function refreshStatus() {
  try { renderSnapshot(await api("/api/status")); }
  catch (error) { toast(`状态连接失败：${error.message}`, true); }
}

async function boot() {
  bindActions();
  try {
    renderConfig(await api("/api/config"));
    await refreshStatus();
    window.setInterval(refreshStatus, 500);
  } catch (error) {
    toast(`UI 初始化失败：${error.message}`, true);
  }
}

window.addEventListener("DOMContentLoaded", boot);
