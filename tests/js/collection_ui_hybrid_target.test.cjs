"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

function selectElement() {
  return {
    children: [],
    disabled: false,
    value: "",
    replaceChildren(...children) {
      this.children = children;
      this.value = children[0]?.value || "";
    },
    appendChild(child) {
      this.children.push(child);
      if (!this.value) this.value = child.value;
    },
  };
}

const groupSelect = selectElement();
const targetSelect = selectElement();
const currentTarget = {textContent: ""};
const hint = {textContent: ""};
const elements = new Map([
  ["hybrid-dig-group", groupSelect],
  ["hybrid-dig-target", targetSelect],
  ["hybrid-target", currentTarget],
  ["hybrid-group-hint", hint],
]);
const context = vm.createContext({
  console,
  document: {
    createElement() { return {textContent: "", value: ""}; },
    getElementById: id => elements.get(id),
  },
  window: {addEventListener() {}},
});
const appJs = path.resolve(
  __dirname,
  "../../src/excavator_il/collection_ui_static/app.js",
);
const appSource = fs.readFileSync(appJs, "utf8");
vm.runInContext(appSource, context, {filename: appJs});

const groups = [
  {group_id: "all", label: "全部 8 点", point_ids: ["near_01", "far_01"]},
  {group_id: "near", label: "近端 4 点", point_ids: ["near_01", "near_02"]},
  {group_id: "far", label: "远端 4 点", point_ids: ["far_01", "far_02", "far_03", "far_04"]},
];
vm.runInContext(`state.config = {hybrid_dig_groups: ${JSON.stringify(groups)}}`, context);

vm.runInContext("renderHybridDigGroups(state.config.hybrid_dig_groups, 'far')", context);
assert.deepEqual(
  targetSelect.children.map(option => option.value),
  ["far_01", "far_02", "far_03", "far_04"],
);
assert.equal(targetSelect.value, "far_01");
assert.equal(vm.runInContext("state.selectedHybridTargetId", context), "far_01");

targetSelect.value = "far_03";
vm.runInContext("selectHybridTarget('far_03')", context);
assert.equal(vm.runInContext("state.selectedHybridTargetId", context), "far_03");
assert.equal(currentTarget.textContent, "far_03");
vm.runInContext(
  "renderHybridTarget({stage: 'idle', dig_target_id: 'dig_01'})",
  context,
);
assert.equal(currentTarget.textContent, "far_03");
vm.runInContext(
  "renderHybridTarget({stage: 'running_rl_to_dig', dig_target_id: 'far_04'})",
  context,
);
assert.equal(currentTarget.textContent, "far_04");

vm.runInContext("selectHybridGroup('near')", context);
assert.deepEqual(
  targetSelect.children.map(option => option.value),
  ["near_01", "near_02"],
);
assert.equal(vm.runInContext("state.selectedHybridTargetId", context), "near_01");
