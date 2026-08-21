"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const cycleSelect = {value: "4"};
const startButton = {textContent: ""};
const elements = new Map([
  ["hybrid-cycle-count", cycleSelect],
  ["hybrid-auto-start", startButton],
]);
const context = vm.createContext({
  console,
  document: {getElementById: id => elements.get(id)},
  window: {addEventListener() {}},
});
const appJs = path.resolve(
  __dirname,
  "../../src/excavator_il/collection_ui_static/app.js",
);
const appSource = fs.readFileSync(appJs, "utf8");
vm.runInContext(appSource, context, {filename: appJs});

assert.match(
  appSource,
  /snapshot\.run_completed_cycles/,
  "the UI progress must use the current run rather than the lifetime total",
);

assert.equal(vm.runInContext("selectedHybridCycleCount()", context), 4);
vm.runInContext("renderHybridCycleButton()", context);
assert.equal(startButton.textContent, "自动装车 4 铲");

cycleSelect.value = "5";
vm.runInContext("renderHybridCycleButton()", context);
assert.equal(startButton.textContent, "自动装车 5 铲");

cycleSelect.value = "999";
assert.equal(
  vm.runInContext("selectedHybridCycleCount()", context),
  4,
  "invalid DOM values must fall back to the commissioned default",
);
