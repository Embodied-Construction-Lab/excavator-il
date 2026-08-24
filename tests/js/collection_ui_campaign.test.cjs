"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const elements = new Map([
  ["task-variant", {value: "dig_only"}],
  ["soil-reset-block-id", {value: "block_01"}],
  ["dig-point-id", {value: "dig_01"}],
  ["campaign-progress", {textContent: ""}],
  ["campaign-next-slot", {textContent: ""}],
  ["dig-target", {textContent: ""}],
  ["hybrid-target", {textContent: ""}],
  ["stage-label", {textContent: ""}],
  ["status-dot", {className: ""}],
  ["start-button", {disabled: false}],
  ["stop-button", {disabled: false}],
]);
const context = vm.createContext({
  console,
  document: {
    getElementById: id => elements.get(id) || null,
    querySelectorAll: () => [],
  },
  window: {addEventListener() {}},
});
const appJs = path.resolve(
  __dirname,
  "../../src/excavator_il/collection_ui_static/app.js",
);
const source = fs.readFileSync(appJs, "utf8");
vm.runInContext(source, context, {filename: appJs});

vm.runInContext(`(() => {
  state.config = {rl_dig_targets: [
    {target_id: "dig_01", position_m: [1.0, 0.0, 0.0]},
    {target_id: "dig_02", position_m: [1.0, 0.2, 0.0]},
  ], campaign_tracking_enabled: true};
  renderCampaignStatus({
    planned: 200,
    completed: 17,
    ignored_diagnostics: 2,
    complete_and_valid: false,
    next_expected_slot: {
      slot_id: "slot_018",
      task_variant: "dig_transport_dump",
      soil_reset_block_id: "block_02",
      dig_point_id: "dig_02",
    },
  });
})()`, context);

assert.equal(elements.get("campaign-progress").textContent, "17 / 200");
assert.equal(elements.get("campaign-next-slot").textContent, "slot_018 · 已忽略诊断 2 条");
assert.equal(elements.get("task-variant").value, "dig_transport_dump");
assert.equal(elements.get("soil-reset-block-id").value, "block_02");
assert.equal(elements.get("dig-point-id").value, "dig_02");
assert.equal(vm.runInContext("state.selectedTargetId", context), "dig_02");
assert.equal(elements.get("start-button").disabled, false);

vm.runInContext('renderCampaignUnavailable("SSH 连接失败")', context);
assert.equal(elements.get("start-button").disabled, true);
vm.runInContext('state.selectedMode = "teleop"; updateOwnershipControls()', context);
assert.equal(elements.get("start-button").disabled, false);

vm.runInContext(`(() => {
  state.selectedMode = "rl";
  state.config.rl_dig_targets = [
    {target_id: "dig_01", position_m: [1.0, 0.0, 0.0]},
  ];
  renderCampaignStatus({
    planned: 200,
    completed: 17,
    ignored_diagnostics: 2,
    complete_and_valid: false,
    next_expected_slot: {
      slot_id: "slot_018",
      task_variant: "dig_transport_dump",
      soil_reset_block_id: "block_02",
      dig_point_id: "dig_02",
    },
  });
})()`, context);
assert.match(elements.get("campaign-next-slot").textContent, /dig_02.*未配置/);
assert.equal(elements.get("start-button").disabled, true);

vm.runInContext(`(() => {
  state.config.rl_dig_targets.push(
    {target_id: "dig_02", position_m: [1.0, 0.2, 0.0]},
  );
  renderCampaignStatus({
    planned: 200,
    completed: 17,
    ignored_diagnostics: 2,
    complete_and_valid: false,
    next_expected_slot: {
      slot_id: "slot_018",
      task_variant: "dig_transport_dump",
      soil_reset_block_id: "block_02",
      dig_point_id: "dig_02",
    },
  });
})()`, context);
assert.equal(elements.get("start-button").disabled, false);
assert.match(source, /setInterval\(refreshCampaignStatus,\s*CAMPAIGN_REFRESH_MS\)/);
