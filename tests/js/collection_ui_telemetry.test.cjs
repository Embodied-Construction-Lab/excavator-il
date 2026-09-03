"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

class FakeClassList {
  constructor() { this.values = new Set(); }
  add(value) { this.values.add(value); }
  remove(value) { this.values.delete(value); }
  toggle(value, enabled) {
    if (enabled) this.add(value);
    else this.remove(value);
  }
  contains(value) { return this.values.has(value); }
}

const telemetryState = {textContent: "", classList: new FakeClassList()};
const metricIds = [
  "angle-boom", "angle-arm", "angle-bucket", "angle-swing",
  "cylinder-boom", "cylinder-stick", "cylinder-bucket",
];
const elements = new Map([["telemetry-state", telemetryState]]);
for (const id of metricIds) elements.set(id, {textContent: "stale"});

const context = vm.createContext({
  console,
  document: {getElementById: id => elements.get(id)},
  window: {addEventListener() {}},
});
const appJs = path.resolve(__dirname, "../../src/excavator_il/collection_ui_static/app.js");
vm.runInContext(fs.readFileSync(appJs, "utf8"), context, {filename: appJs});

vm.runInContext(`renderTelemetry({
  source: "machine_state_v1/udp:18081",
  age_ms: 12,
  sensor_valid: true,
  control_enabled: true,
  fault_flags: [],
  joint_angles_deg: {boom: 1, arm: 2, bucket: 3, swing: 4},
  cylinders_mm: {boom: 101, stick: 202, bucket: 303},
})`, context);

assert.equal(telemetryState.textContent, "实时 · UDP 18081 · 12 ms");
assert.equal(elements.get("angle-swing").textContent, "4.0");

vm.runInContext("renderTelemetryUnavailable()", context);

assert.equal(telemetryState.textContent, "等待机器状态 · UDP 18081");
for (const id of metricIds) assert.equal(elements.get(id).textContent, "—");
