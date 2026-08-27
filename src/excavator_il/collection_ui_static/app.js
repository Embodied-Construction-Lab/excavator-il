"use strict";

const UI_HEADER = {"X-Excavator-UI": "1"};
const HYBRID_MOTION_AUTHORIZATION = "ALLOW_HYBRID_MACHINE_MOTION";
const terminalStages = new Set(["idle", "completed", "failed", "cancelled"]);
const hybridTerminalStages = new Set(["idle", "completed", "failed", "cancelled"]);
const stageLabels = {
  idle: "空闲", starting: "启动中", preflight: "环境检查",
  rl_positioning: "RL 定位", collector_starting: "Collector 启动",
  manual_positioning: "手工预定位", teleoperation: "仅遥操作",
  recorder_standby: "等待 deadman",
  recording: "记录中", review: "结果确认", finalizing: "保存中",
  validating: "校验中", completed: "已完成", stopping: "停止中",
  failed: "失败", cancelled: "已取消"
};
const progressOrder = ["position", "standby", "record", "review", "done"];
const stageProgress = {
  rl_positioning: 0, collector_starting: 0, manual_positioning: 0,
  teleoperation: 0,
  recorder_standby: 1, recording: 2, review: 3,
  finalizing: 4, validating: 4, completed: 4
};
const hybridStageLabels = {
  idle: "空闲", starting: "启动中", running_rl_to_dig: "RL 到挖点",
  awaiting_act_dig: "等待 ACT 挖掘", running_act_dig: "ACT 挖掘中",
  awaiting_rl_to_dump: "等待前往倾倒点",
  running_rl_to_dump_and_dump: "RL 到倾倒点并倾倒",
  awaiting_rl_return: "等待返回挖点", running_rl_return_to_dig: "RL 返回挖点",
  completed: "装车任务完成", stopping: "安全停止中", failed: "失败", cancelled: "已取消"
};
const hybridSegments = ["rl_to_dig", "act_dig", "rl_to_dump_and_dump", "rl_return_to_dig"];

const state = {
  selectedMode: "rl",
  selectedTargetId: null,
  config: null,
  snapshot: null,
  hybridSnapshot: null,
  operatorSnapshot: null,
  cameraStreams: {}
};
const $ = (id) => document.getElementById(id);
const CAMERA_RETRY_MS = 1000;
const CAMERA_REFRESH_MS = 100;
const LOG_BOTTOM_TOLERANCE_PX = 24;
const logAutofollow = new Map();

function logIsNearBottom(node) {
  return node.scrollHeight - node.scrollTop - node.clientHeight <= LOG_BOTTOM_TOLERANCE_PX;
}

function bindLogPanel(id) {
  const node = $(id);
  if (!node || logAutofollow.has(id)) return;
  logAutofollow.set(id, true);
  node.addEventListener("scroll", () => {
    logAutofollow.set(id, logIsNearBottom(node));
  });
}

function renderLogContent(id, lines, emptyText) {
  const node = $(id);
  if (!node) return;
  const shouldFollow = logAutofollow.get(id) !== false;
  node.textContent = lines.length ? lines.join("\n") : emptyText;
  if (shouldFollow) {
    const schedule = window.requestAnimationFrame || (callback => callback());
    schedule(() => {
      node.scrollTop = node.scrollHeight;
    });
  }
}

async function copyLogContent(id) {
  const node = $(id);
  if (!node) throw new Error("日志区域不存在");
  const content = node.textContent || "";
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(content);
  } else {
    const textarea = document.createElement("textarea");
    textarea.value = content;
    textarea.setAttribute("readonly", "");
    textarea.style.position = "fixed";
    textarea.style.opacity = "0";
    document.body.appendChild(textarea);
    textarea.select();
    const copied = document.execCommand("copy");
    textarea.remove();
    if (!copied) throw new Error("浏览器拒绝复制日志");
  }
}

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
  renderTargets(config.rl_dig_targets || []);
  if (config.hybrid_mission_enabled) {
    $("hybrid-panel").classList.remove("hidden");
    $("hybrid-act-steps").textContent = String(config.hybrid_act_max_steps);
    const localCycle = config.hybrid_runtime_backend === "resident_fixed_cycle";
    $("hybrid-segmented-start").classList.toggle("hidden", localCycle);
    $("hybrid-advance").classList.toggle("hidden", localCycle);
    if (localCycle) {
      $("hybrid-runtime-description").textContent =
        "V3-A：PC 只启动/取消/显示，固定点轨迹与 RL/ACT 切换均在 Orin 本地完成。";
    }
  }
  if (config.operator_control_enabled) {
    $("operator-control").classList.remove("hidden");
  }
  const previewUrls = config.camera_preview_urls || {
    front: config.camera_preview_url
  };
  setupCameraPreview("front", previewUrls.front);
  if (previewUrls.dump) {
    $("camera-dump-container")?.classList.remove("hidden");
    setupCameraPreview("dump", previewUrls.dump);
  }
  renderTaskRecordingHint();
}

function cameraElements(cameraId) {
  const prefix = cameraId === "front" ? "camera" : `camera-${cameraId}`;
  return {
    image: $(`${prefix}-preview`),
    placeholder: $(`${prefix}-placeholder`),
    status: $(`${prefix}-state`)
  };
}

function setupCameraPreview(cameraId, url) {
  if (!url) return;
  const elements = cameraElements(cameraId);
  if (!elements.image || !elements.placeholder || !elements.status) return;
  state.cameraStreams[cameraId] = {url, retryTimer: null, attempt: 0};
  const {image, placeholder, status} = elements;
  image.addEventListener("load", () => {
    const stream = state.cameraStreams[cameraId];
    if (stream.retryTimer !== null) {
      window.clearTimeout(stream.retryTimer);
      stream.retryTimer = null;
    }
    image.classList.add("ready");
    placeholder.classList.add("hidden");
    status.textContent = `实时 · 帧 ${stream.attempt}`;
    scheduleCameraRefresh(cameraId, CAMERA_REFRESH_MS);
  });
  image.addEventListener("error", () => {
    image.classList.remove("ready");
    placeholder.classList.remove("hidden");
    status.textContent = "等待 Collector";
    scheduleCameraRefresh(cameraId, CAMERA_RETRY_MS);
  });
  loadCameraPreview(cameraId);
}

function renderTargets(targets) {
  const grid = $("target-grid");
  if (!grid) return;
  grid.replaceChildren();
  const digPoint = $("dig-point-id");
  if (digPoint && targets.length) digPoint.replaceChildren();
  state.selectedTargetId = targets[0]?.target_id || digPoint?.value || null;
  $("target-count").textContent = `${targets.length} 个可选点`;
  targets.forEach((target, index) => {
    if (digPoint) {
      const option = document.createElement("option");
      option.value = target.target_id;
      option.textContent = target.target_id;
      digPoint.appendChild(option);
    }
    const button = document.createElement("button");
    button.type = "button";
    button.className = `target-card${index === 0 ? " selected" : ""}`;
    button.dataset.targetId = target.target_id;
    const coordinates = target.position_m.map(value => Number(value).toFixed(2)).join(", ");
    const label = document.createElement("strong");
    label.textContent = target.target_id;
    const detail = document.createElement("small");
    detail.textContent = `${coordinates} m`;
    button.append(label, detail);
    button.addEventListener("click", () => selectTarget(target));
    grid.appendChild(button);
  });
  if (targets[0]) selectTarget(targets[0]);
}

function selectTarget(target) {
  state.selectedTargetId = target.target_id;
  document.querySelectorAll(".target-card").forEach(card => {
    card.classList.toggle("selected", card.dataset.targetId === target.target_id);
  });
  $("dig-target").textContent = target.position_m
    .map(value => Number(value).toFixed(2)).join(", ");
  if ($("hybrid-target")) $("hybrid-target").textContent = target.target_id;
  if ($("dig-point-id")) $("dig-point-id").value = target.target_id;
}

function renderSelectedMode() {
  const isRl = state.selectedMode === "rl";
  const isTeleop = state.selectedMode === "teleop";
  $("rl-target-section").classList.toggle("hidden", !isRl);
  $("collection-protocol-panel")?.classList.toggle("hidden", isTeleop);
  $("collection-timeline").classList.toggle("hidden", isTeleop);
  $("start-button").textContent = isTeleop ? "启动仅遥操作" : "开始采集流程";
  $("batch-hint").textContent = isTeleop
    ? "仅遥操作不会启动 Recorder 或创建 Episode；点击安全停止后释放串口和相机。"
    : "每条完成后可直接选择下一点并再次开始；计数只统计本次 UI 运行期间成功完成校验的 Episode。";
  renderTaskRecordingHint();
  if (!isRl) {
    $("dig-target").textContent = state.config.dig_target_m
      .map(value => Number(value).toFixed(2)).join(", ");
    updateOwnershipControls();
    return;
  }
  const selected = (state.config.rl_dig_targets || [])
    .find(target => target.target_id === state.selectedTargetId);
  if (selected) selectTarget(selected);
  updateOwnershipControls();
}

function renderTaskRecordingHint() {
  const hint = $("task-recording-hint");
  if (!hint) return;
  hint.textContent = $("task-variant")?.value === "dig_transport_dump"
    ? "连续完成挖掘、提升、回转和倾倒；动作全部结束并稳定后再松开 deadman。"
    : "完成挖掘与提升并稳定后松开 deadman；不要录入后续回转和倾倒。";
}

function loadCameraPreview(cameraId = "front") {
  const stream = state.cameraStreams[cameraId];
  if (!stream?.url) return;
  stream.attempt += 1;
  const separator = stream.url.includes("?") ? "&" : "?";
  cameraElements(cameraId).image.src = `${stream.url}${separator}frame=${stream.attempt}`;
}

function scheduleCameraRefresh(cameraId, delayMs) {
  const stream = state.cameraStreams[cameraId];
  if (!stream || stream.retryTimer !== null) return;
  stream.retryTimer = window.setTimeout(() => {
    stream.retryTimer = null;
    loadCameraPreview(cameraId);
  }, delayMs);
}

function renderSnapshot(snapshot) {
  state.snapshot = snapshot;
  const stage = snapshot.stage || "idle";
  const active = !terminalStages.has(stage);
  if (active && snapshot.positioning_mode) {
    state.selectedMode = snapshot.positioning_mode;
    document.querySelectorAll(".mode-card").forEach(card => {
      card.classList.toggle("selected", card.dataset.mode === state.selectedMode);
    });
    renderSelectedMode();
  }
  $("stage-label").textContent = stageLabels[stage] || stage;
  $("status-dot").className = `status-dot${active ? " active" : ""}${stage === "failed" ? " error" : ""}`;

  const manual = stage === "manual_positioning";
  const review = stage === "review";
  $("stage-actions").classList.toggle("hidden", !manual && !review);
  $("manual-actions").classList.toggle("hidden", !manual);
  $("review-actions").classList.toggle("hidden", !review);

  const logs = Array.isArray(snapshot.logs) ? snapshot.logs : [];
  const protocolLine = collectionProtocolLine(snapshot);
  const visibleLogs = protocolLine && !logs.includes(protocolLine)
    ? [protocolLine, ...logs]
    : logs;
  renderLogContent("log-output", visibleLogs, "等待采集任务…");
  if (snapshot.task_variant) $("task-variant").value = snapshot.task_variant;
  if (snapshot.soil_reset_block_id) $("soil-reset-block-id").value = snapshot.soil_reset_block_id;
  if (snapshot.dig_point_id) $("dig-point-id").value = snapshot.dig_point_id;
  if (snapshot.collection_zone_id) $("collection-zone-id").value = snapshot.collection_zone_id;
  if (snapshot.dig_repeat_index) $("dig-repeat-index").value = String(snapshot.dig_repeat_index);
  if (terminalStages.has(stage)) {
    $("operator-note").value = "";
  } else if (snapshot.operator_note !== undefined) {
    $("operator-note").value = snapshot.operator_note;
  }
  if ($("episode-context")) {
    $("episode-context").textContent = protocolLine.replace("[episode-context] ", "");
  }
  $("episode-path").textContent = snapshot.episode_path || "";
  $("episode-path").title = snapshot.episode_path || "";
  $("error-banner").textContent = snapshot.error || "";
  $("error-banner").classList.toggle("hidden", !snapshot.error);
  renderProgress(stage);
  updateOwnershipControls();
}

function renderHybridSnapshot(snapshot) {
  state.hybridSnapshot = snapshot;
  const stage = snapshot.stage || "idle";
  $("hybrid-stage").textContent = hybridStageLabels[stage] || stage;
  $("hybrid-target").textContent = snapshot.dig_target_id || state.selectedTargetId || "—";
  const requestedCycles = Number(snapshot.requested_cycles || 1);
  $("hybrid-cycles").textContent = `${snapshot.run_completed_cycles || 0} / ${requestedCycles} 铲`;
  const logs = Array.isArray(snapshot.logs) ? snapshot.logs : [];
  renderLogContent("hybrid-log", logs, "等待混合 Mission…");
  $("hybrid-error").textContent = snapshot.error || "";
  $("hybrid-error").classList.toggle("hidden", !snapshot.error);
  const runningSegment = stage.startsWith("running_") ? stage.slice("running_".length) : "";
  const nextSegment = snapshot.next_segment || "";
  const currentIndex = hybridSegments.indexOf(runningSegment || nextSegment);
  document.querySelectorAll("[data-hybrid-segment]").forEach(node => {
    const index = hybridSegments.indexOf(node.dataset.hybridSegment);
    node.classList.toggle("active", index === currentIndex && stage !== "completed");
    node.classList.toggle("done", stage === "completed" || (currentIndex >= 0 && index < currentIndex));
  });
  $("hybrid-advance").textContent = nextSegment === "act_dig"
    ? "执行 ACT 挖掘"
    : nextSegment === "rl_to_dump_and_dump"
      ? "前往倾倒并倾倒"
      : nextSegment === "rl_return_to_dig" ? "RL 返回挖点" : "执行下一段";
  updateOwnershipControls();
}

function renderOperatorSnapshot(snapshot) {
  state.operatorSnapshot = snapshot;
  const labels = {
    stopped: "未启动", starting: "启动中", ready: "已就绪", failed: "启动失败"
  };
  $("operator-stage").textContent = labels[snapshot.stage] || snapshot.stage;
  updateOwnershipControls();
}

function updateOwnershipControls() {
  const collectionStage = state.snapshot?.stage || "idle";
  const hybridStage = state.hybridSnapshot?.stage || "idle";
  const collectionActive = !terminalStages.has(collectionStage);
  const hybridActive = !hybridTerminalStages.has(hybridStage);
  const hybridCanStop = state.hybridSnapshot?.can_stop === true;
  if (hybridActive) {
    $("stage-label").textContent = `闭环 · ${hybridStageLabels[hybridStage] || hybridStage}`;
    $("status-dot").className = `status-dot active${hybridStage === "failed" ? " error" : ""}`;
  } else {
    $("stage-label").textContent = stageLabels[collectionStage] || collectionStage;
    $("status-dot").className = `status-dot${collectionActive ? " active" : ""}${collectionStage === "failed" ? " error" : ""}`;
  }
  $("start-button").disabled = collectionActive || hybridActive;
  $("stop-button").disabled = !collectionActive || collectionStage === "stopping";
  document.querySelectorAll(".mode-card").forEach(card => { card.disabled = collectionActive || hybridActive; });
  document.querySelectorAll(".target-card").forEach(card => { card.disabled = collectionActive || hybridActive; });
  [
    "task-variant",
    "soil-reset-block-id",
    "dig-point-id",
    "collection-zone-id",
    "dig-repeat-index",
    "operator-note",
  ].forEach(id => {
    if ($(id)) $(id).disabled = collectionActive || hybridActive;
  });
  if (!state.config?.hybrid_mission_enabled) return;
  $("hybrid-segmented-start").disabled = collectionActive || hybridActive;
  $("hybrid-auto-start").disabled = collectionActive || hybridActive;
  $("hybrid-cycle-count").disabled = collectionActive || hybridActive;
  $("hybrid-advance").disabled = collectionActive || !hybridStage.startsWith("awaiting_");
  $("hybrid-stop").disabled = !hybridCanStop || hybridStage === "stopping";
  if (state.config.operator_control_enabled) {
    const operatorStage = state.operatorSnapshot?.stage || "stopped";
    const operatorActive = operatorStage === "starting" || operatorStage === "ready";
    $("operator-start").disabled = collectionActive || hybridActive || operatorActive;
    $("operator-stop").disabled = collectionActive || hybridActive || operatorStage === "stopped";
  }
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
  bindLogPanel("log-output");
  bindLogPanel("hybrid-log");
  $("copy-log")?.addEventListener("click", () => {
    copyLogContent("log-output")
      .then(() => toast("采集日志已复制"))
      .catch(error => toast(error.message, true));
  });
  $("copy-hybrid-log")?.addEventListener("click", () => {
    copyLogContent("hybrid-log")
      .then(() => toast("Mission 日志已复制"))
      .catch(error => toast(error.message, true));
  });
  $("clear-log")?.addEventListener("click", () => {
    command("/api/collection/logs/clear");
  });
  $("clear-hybrid-log")?.addEventListener("click", () => {
    commandHybrid("/api/hybrid/logs/clear");
  });
  document.querySelectorAll(".mode-card").forEach(card => card.addEventListener("click", () => {
    state.selectedMode = card.dataset.mode;
    renderSelectedMode();
    document.querySelectorAll(".mode-card").forEach(node => node.classList.toggle("selected", node === card));
  }));
  $("dig-point-id")?.addEventListener("change", event => {
    const target = (state.config?.rl_dig_targets || [])
      .find(candidate => candidate.target_id === event.target.value);
    if (target) selectTarget(target);
  });
  $("task-variant")?.addEventListener("change", renderTaskRecordingHint);
  $("collection-zone-id")?.addEventListener("change", updateReferenceDigPoint);
  $("start-button").addEventListener("click", () => command(
    "/api/collection/start",
    collectionStartPayload(),
  ));
  $("stop-button").addEventListener("click", () => command("/api/collection/stop"));
  $("manual-complete-button").addEventListener("click", () => command("/api/collection/manual-complete"));
  document.querySelectorAll("[data-outcome]").forEach(button => button.addEventListener("click", () => {
    command("/api/collection/outcome", {outcome: button.dataset.outcome});
  }));
  $("hybrid-segmented-start").addEventListener("click", () => commandHybrid("/api/hybrid/start", {
    dig_target_id: state.selectedTargetId,
    automatic: false,
    cycle_count: 1,
    motion_authorization: null
  }));
  $("hybrid-cycle-count").addEventListener("change", renderHybridCycleButton);
  $("hybrid-auto-start").addEventListener("click", () => {
    commandHybrid("/api/hybrid/start", {
      dig_target_id: state.selectedTargetId,
      automatic: true,
      cycle_count: selectedHybridCycleCount(),
      motion_authorization: hybridMotionAuthorization()
    });
  });
  $("hybrid-advance").addEventListener("click", () => {
    const needsAuthorization = state.hybridSnapshot?.next_segment === "act_dig";
    const authorization = needsAuthorization
      ? hybridMotionAuthorization()
      : null;
    commandHybrid("/api/hybrid/advance", {motion_authorization: authorization});
  });
  $("hybrid-stop").addEventListener("click", () => commandHybrid("/api/hybrid/stop"));
  $("operator-start").addEventListener("click", () => commandOperator("/api/operator/start"));
  $("operator-stop").addEventListener("click", () => commandOperator("/api/operator/stop"));
}

function collectionStartPayload() {
  const teleop = state.selectedMode === "teleop";
  updateReferenceDigPoint();
  return {
    positioning_mode: state.selectedMode,
    dig_target_id: state.selectedMode === "rl" ? state.selectedTargetId : null,
    task_variant: teleop ? null : $("task-variant")?.value || null,
    soil_reset_block_id: teleop ? null : $("soil-reset-block-id")?.value || null,
    dig_point_id: teleop ? null : $("dig-point-id")?.value || state.selectedTargetId,
    collection_zone_id: teleop ? null : $("collection-zone-id")?.value || null,
    dig_repeat_index: teleop
      ? null
      : Number.parseInt($("dig-repeat-index")?.value || "", 10),
    operator_note: teleop ? null : $("operator-note")?.value?.trim() || "",
  };
}

function updateReferenceDigPoint() {
  if (state.selectedMode === "rl") return;
  const zoneId = $("collection-zone-id")?.value || "zone_01";
  const zoneNumber = Number.parseInt(zoneId.slice(-2), 10);
  const column = Number.isInteger(zoneNumber) ? ((zoneNumber - 1) % 3) + 1 : 1;
  const targetId = `dig_${String(column).padStart(2, "0")}`;
  const configured = (state.config?.rl_dig_targets || [])
    .some(target => target.target_id === targetId);
  if (configured && $("dig-point-id")) $("dig-point-id").value = targetId;
}

function collectionProtocolLine(snapshot) {
  if (!snapshot.task_variant) return "";
  return "[episode-context] "
    + `task_variant=${snapshot.task_variant} `
    + `soil_reset_block_id=${snapshot.soil_reset_block_id} `
    + `dig_point_id=${snapshot.dig_point_id} `
    + `collection_zone_id=${snapshot.collection_zone_id} `
    + `dig_repeat_index=${snapshot.dig_repeat_index}`
    + (snapshot.operator_note ? ` operator_note=${snapshot.operator_note}` : "");
}

function hybridMotionAuthorization() {
  return HYBRID_MOTION_AUTHORIZATION;
}

function selectedHybridCycleCount() {
  const value = Number.parseInt($("hybrid-cycle-count")?.value || "4", 10);
  return Number.isInteger(value) && value >= 1 && value <= 9 ? value : 4;
}

function renderHybridCycleButton() {
  $("hybrid-auto-start").textContent = `自动装车 ${selectedHybridCycleCount()} 铲`;
}

async function commandHybrid(path, body) {
  const headers = {...UI_HEADER};
  const options = {method: "POST", headers};
  if (body !== undefined) {
    headers["Content-Type"] = "application/json";
    options.body = JSON.stringify(body);
  }
  try { renderHybridSnapshot(await api(path, options)); }
  catch (error) { toast(error.message, true); }
}

async function commandOperator(path) {
  try {
    renderOperatorSnapshot(await api(path, {method: "POST", headers: UI_HEADER}));
  } catch (error) {
    toast(error.message, true);
    await refreshOperatorStatus();
  }
}

function setMetric(id, value, digits = 1) {
  const node = $(id);
  if (node) node.textContent = Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : "—";
}

function renderTelemetry(payload) {
  const ageMs = Number(payload.age_ms);
  const fresh = Number.isFinite(ageMs) && ageMs <= 250;
  const hardwareFaults = Number(payload.fault_flags || 0) & ~16;
  const healthy = payload.sensor_valid === true && hardwareFaults === 0;
  const badge = $("telemetry-state");
  badge.textContent = !fresh
    ? "遥测过期"
    : !healthy ? "传感器 / 硬件故障"
    : payload.control_enabled === true
      ? `实时 · ${ageMs.toFixed(0)} ms`
      : "实时 · 安全零位";
  badge.classList.toggle("waiting", !fresh || !healthy);
  const angles = payload.joint_angles_deg || {};
  const cylinders = payload.cylinders_mm || {};
  setMetric("angle-boom", angles.boom);
  setMetric("angle-arm", angles.arm);
  setMetric("angle-bucket", angles.bucket);
  setMetric("angle-swing", angles.swing);
  setMetric("cylinder-boom", cylinders.boom);
  setMetric("cylinder-stick", cylinders.stick);
  setMetric("cylinder-bucket", cylinders.bucket);
}

function renderTelemetryUnavailable() {
  const badge = $("telemetry-state");
  if (badge) {
    badge.textContent = "等待 Collector";
    badge.classList.add("waiting");
  }
}

async function refreshTelemetry() {
  try { renderTelemetry(await api("/api/telemetry")); }
  catch (_error) { renderTelemetryUnavailable(); }
}

async function refreshStatus() {
  try { renderSnapshot(await api("/api/status")); }
  catch (error) { toast(`状态连接失败：${error.message}`, true); }
}

async function refreshHybridStatus() {
  if (!state.config?.hybrid_mission_enabled) return;
  try { renderHybridSnapshot(await api("/api/hybrid/status")); }
  catch (error) { toast(`混合 Mission 状态失败：${error.message}`, true); }
}

async function refreshOperatorStatus() {
  if (!state.config?.operator_control_enabled) return;
  try { renderOperatorSnapshot(await api("/api/operator/status")); }
  catch (error) { toast(`RL/RViz 状态失败：${error.message}`, true); }
}

async function boot() {
  bindActions();
  renderHybridCycleButton();
  try {
    renderConfig(await api("/api/config"));
    await refreshStatus();
    window.setInterval(refreshStatus, 500);
    await refreshHybridStatus();
    window.setInterval(refreshHybridStatus, 500);
    await refreshOperatorStatus();
    window.setInterval(refreshOperatorStatus, 1000);
    await refreshTelemetry();
    window.setInterval(refreshTelemetry, 500);
  } catch (error) {
    toast(`UI 初始化失败：${error.message}`, true);
  }
}

window.addEventListener("DOMContentLoaded", boot);
