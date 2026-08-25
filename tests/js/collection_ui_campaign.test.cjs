"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const elements = new Map([
  ["task-variant", {value: "dig_transport_dump", disabled: false}],
  ["soil-reset-block-id", {value: "soil_after_rain", disabled: false}],
  ["dig-point-id", {value: "dig_03", disabled: false}],
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
  state.config = {hybrid_mission_enabled: false};
  state.selectedMode = "direct";
  updateOwnershipControls();
})()`, context);

assert.equal(elements.get("start-button").disabled, false);
assert.equal(elements.get("task-variant").value, "dig_transport_dump");
assert.equal(elements.get("soil-reset-block-id").value, "soil_after_rain");
assert.equal(elements.get("dig-point-id").value, "dig_03");
assert.doesNotMatch(source, /\/api\/campaign\/status/);
assert.doesNotMatch(source, /renderCampaignStatus/);
assert.doesNotMatch(source, /CAMPAIGN_REFRESH_MS/);
