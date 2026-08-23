"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

class FakeClassList {
  constructor() { this.values = new Set(["hidden"]); }
  add(value) { this.values.add(value); }
  remove(value) { this.values.delete(value); }
  contains(value) { return this.values.has(value); }
}

class FakeImage {
  constructor() {
    this.classList = new FakeClassList();
    this.listeners = new Map();
    this.assignments = [];
  }
  addEventListener(name, callback) { this.listeners.set(name, callback); }
  set src(value) { this.assignments.push(value); }
  dispatch(name) { this.listeners.get(name)?.(); }
}

const front = new FakeImage();
const dump = new FakeImage();
const frontPlaceholder = {classList: new FakeClassList()};
const dumpPlaceholder = {classList: new FakeClassList()};
const frontState = {textContent: ""};
const dumpState = {textContent: ""};
const dumpContainer = {classList: new FakeClassList()};
const elements = new Map([
  ["camera-preview", front],
  ["camera-placeholder", frontPlaceholder],
  ["camera-state", frontState],
  ["camera-dump-preview", dump],
  ["camera-dump-placeholder", dumpPlaceholder],
  ["camera-dump-state", dumpState],
  ["camera-dump-container", dumpContainer],
  ["operator-id", {textContent: ""}],
  ["task-name", {textContent: ""}],
  ["dig-target", {textContent: ""}],
  ["orin-host", {textContent: ""}],
]);
const timers = [];

const context = vm.createContext({
  console,
  document: {getElementById: id => elements.get(id)},
  window: {
    addEventListener() {},
    setTimeout(callback) { timers.push(callback); return timers.length; },
    clearTimeout() {},
  },
});
const appJs = path.resolve(__dirname, "../../src/excavator_il/collection_ui_static/app.js");
vm.runInContext(fs.readFileSync(appJs, "utf8"), context, {filename: appJs});

vm.runInContext(`renderConfig({
  operator_id: "zhaoshuai",
  task: "ExecuteDigAndDump",
  dig_target_m: [0.8, 0.0, -0.2],
  orin_host: "192.168.50.2",
  camera_preview_url: "/api/camera/frame.jpg",
  camera_preview_urls: {
    front: "/api/camera/front.jpg",
    dump: "/api/camera/dump.jpg"
  },
  visualization_url: ""
})`, context);

assert.match(front.assignments[0], /^\/api\/camera\/front\.jpg\?frame=1$/);
assert.match(dump.assignments[0], /^\/api\/camera\/dump\.jpg\?frame=1$/);
assert.equal(dumpContainer.classList.contains("hidden"), false);

front.dispatch("load");
dump.dispatch("error");
assert.equal(frontState.textContent, "实时 · 帧 1");
assert.equal(frontPlaceholder.classList.contains("hidden"), true);
assert.equal(dumpState.textContent, "等待 Collector");
assert.equal(dumpPlaceholder.classList.contains("hidden"), false);
assert.equal(timers.length, 2, "each camera owns an independent refresh timer");
