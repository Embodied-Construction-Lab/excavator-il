"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const elements = new Map([
  ["task-variant", {value: "dig_transport_dump"}],
  ["soil-reset-block-id", {value: "block_12"}],
  ["dig-point-id", {value: "dig_03"}],
  ["collection-zone-id", {value: "zone_06"}],
  ["dig-repeat-index", {value: "3"}],
  ["operator-note", {value: "远排右侧第三次"}],
]);
const context = vm.createContext({
  console,
  document: {getElementById: id => elements.get(id) || null},
  window: {addEventListener() {}},
});
const appJs = path.resolve(__dirname, "../../src/excavator_il/collection_ui_static/app.js");
vm.runInContext(fs.readFileSync(appJs, "utf8"), context, {filename: appJs});

const payload = vm.runInContext(`(() => {
  state.selectedMode = "rl";
  state.selectedTargetId = "dig_03";
  return collectionStartPayload();
})()`, context);
assert.deepEqual(JSON.parse(JSON.stringify(payload)), {
  positioning_mode: "rl",
  dig_target_id: "dig_03",
  task_variant: "dig_transport_dump",
  soil_reset_block_id: "block_12",
  dig_point_id: "dig_03",
  collection_zone_id: "zone_06",
  dig_repeat_index: 3,
  operator_note: "远排右侧第三次",
});

const contextLine = vm.runInContext(`collectionProtocolLine({
  task_variant: "dig_transport_dump",
  soil_reset_block_id: "block_12",
  dig_point_id: "dig_03",
  collection_zone_id: "zone_06",
  dig_repeat_index: 3,
  operator_note: "远排右侧第三次"
})`, context);
assert.equal(
  contextLine,
  "[episode-context] task_variant=dig_transport_dump soil_reset_block_id=block_12 dig_point_id=dig_03 collection_zone_id=zone_06 dig_repeat_index=3 operator_note=远排右侧第三次",
);
