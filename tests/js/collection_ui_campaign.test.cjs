"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const elements = new Map([
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
assert.doesNotMatch(source, /task-variant/);
assert.doesNotMatch(source, /soil-reset-block-id/);
assert.doesNotMatch(source, /dig-point-id/);
assert.doesNotMatch(source, /\/api\/campaign\/status/);
assert.doesNotMatch(source, /renderCampaignStatus/);
assert.doesNotMatch(source, /CAMPAIGN_REFRESH_MS/);
