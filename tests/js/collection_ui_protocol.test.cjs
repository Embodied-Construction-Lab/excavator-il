"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const elements = new Map();
const context = vm.createContext({
  console,
  document: {getElementById: id => elements.get(id) || null},
  window: {addEventListener() {}},
});
const appJs = path.resolve(__dirname, "../../src/excavator_il/collection_ui_static/app.js");
vm.runInContext(fs.readFileSync(appJs, "utf8"), context, {filename: appJs});

const payload = vm.runInContext(`(() => {
  state.selectedMode = "rl";
  state.selectedTargetId = "dig_far_04";
  return collectionStartPayload();
})()`, context);
assert.deepEqual(JSON.parse(JSON.stringify(payload)), {
  positioning_mode: "rl",
  dig_target_id: "dig_far_04",
});
