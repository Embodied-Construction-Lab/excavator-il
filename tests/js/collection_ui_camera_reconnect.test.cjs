"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

class FakeClassList {
  constructor() { this.values = new Set(); }
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

const image = new FakeImage();
const placeholder = {classList: new FakeClassList()};
const cameraState = {textContent: ""};
const elements = new Map([
  ["camera-preview", image],
  ["camera-placeholder", placeholder],
  ["camera-state", cameraState],
  ["operator-id", {textContent: ""}],
  ["task-name", {textContent: ""}],
  ["dig-target", {textContent: ""}],
  ["orin-host", {textContent: ""}],
]);
const timers = [];

const context = vm.createContext({
  console,
  document: {
    getElementById: id => elements.get(id),
  },
  window: {
    addEventListener() {},
    setTimeout(callback) {
      timers.push(callback);
      return timers.length;
    },
    clearTimeout() {},
  },
});
const appJs = path.resolve(__dirname, "../../src/excavator_il/collection_ui_static/app.js");
vm.runInContext(fs.readFileSync(appJs, "utf8"), context, {filename: appJs});

vm.runInContext(`renderConfig({
  operator_id: "zhaoshuai",
  task: "ExecuteDig",
  dig_target_m: [0.8, 0.0, -0.2],
  orin_host: "192.168.50.2",
  camera_preview_url: "http://192.168.50.2:18092/camera/front.mjpg",
  visualization_url: ""
})`, context);

assert.equal(image.assignments.length, 1);
image.dispatch("error");
assert.equal(cameraState.textContent, "等待 Collector");
assert.equal(timers.length, 1, "a failed preview should schedule reconnection");

timers.shift()();
assert.equal(image.assignments.length, 2, "reconnection should reload the preview URL");
assert.notEqual(image.assignments[1], image.assignments[0], "retry URL should bypass caches");

image.dispatch("load");
assert.equal(cameraState.textContent, "实时");
assert.equal(image.classList.contains("ready"), true);
assert.equal(placeholder.classList.contains("hidden"), true);
