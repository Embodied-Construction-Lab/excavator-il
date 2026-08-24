"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

class FakeLog {
  constructor() {
    this.textContent = "";
    this.scrollTop = 0;
    this.scrollHeight = 400;
    this.clientHeight = 100;
    this.listeners = new Map();
  }

  addEventListener(name, callback) { this.listeners.set(name, callback); }
  dispatch(name) { this.listeners.get(name)?.(); }
}

const log = new FakeLog();
const copied = [];
const elements = new Map([["hybrid-log", log]]);
const context = vm.createContext({
  console,
  document: {getElementById: id => elements.get(id)},
  navigator: {clipboard: {writeText: async text => copied.push(text)}},
  window: {
    addEventListener() {},
    requestAnimationFrame(callback) { callback(); },
  },
});
const appJs = path.resolve(
  __dirname,
  "../../src/excavator_il/collection_ui_static/app.js",
);
vm.runInContext(fs.readFileSync(appJs, "utf8"), context, {filename: appJs});

vm.runInContext('bindLogPanel("hybrid-log")', context);
log.scrollTop = 300;
vm.runInContext(
  'renderLogContent("hybrid-log", ["first", "second"], "empty")',
  context,
);
assert.equal(log.textContent, "first\nsecond");
assert.equal(log.scrollTop, 400, "a log already at the bottom should follow output");

log.scrollHeight = 500;
log.scrollTop = 20;
log.dispatch("scroll");
log.scrollHeight = 600;
vm.runInContext(
  'renderLogContent("hybrid-log", ["first", "second", "third"], "empty")',
  context,
);
assert.equal(log.scrollTop, 20, "manual upward scrolling must not be overridden");

log.scrollTop = 500;
log.dispatch("scroll");
log.scrollHeight = 700;
vm.runInContext(
  'renderLogContent("hybrid-log", ["first", "second", "third", "fourth"], "empty")',
  context,
);
assert.equal(log.scrollTop, 700, "returning to the bottom should restore auto-follow");

(async () => {
  await vm.runInContext('copyLogContent("hybrid-log")', context);
  assert.deepEqual(copied, ["first\nsecond\nthird\nfourth"]);
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
