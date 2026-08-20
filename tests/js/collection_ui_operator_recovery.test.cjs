"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const nodes = new Map();
function node(id) {
  if (!nodes.has(id)) {
    nodes.set(id, {
      id,
      disabled: false,
      textContent: "",
      className: "",
      classList: {toggle() {}},
      addEventListener() {},
    });
  }
  return nodes.get(id);
}

const context = vm.createContext({
  console,
  document: {
    getElementById(id) { return node(id); },
    querySelectorAll() { return []; },
  },
  window: {addEventListener() {}},
});
const appJs = path.resolve(
  __dirname,
  "../../src/excavator_il/collection_ui_static/app.js",
);
vm.runInContext(fs.readFileSync(appJs, "utf8"), context, {filename: appJs});
vm.runInContext(`
  state.config = {hybrid_mission_enabled: true, operator_control_enabled: true};
  state.snapshot = {stage: "idle"};
  state.hybridSnapshot = {stage: "idle"};
  state.operatorSnapshot = {stage: "failed"};
  updateOwnershipControls();
`, context);

assert.equal(node("operator-start").disabled, false);
assert.equal(node("operator-stop").disabled, false);
