"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

let promptCalls = 0;
const context = vm.createContext({
  console,
  document: {
    getElementById() { return undefined; },
  },
  window: {
    addEventListener() {},
    prompt() {
      promptCalls += 1;
      throw new Error("motion authorization must not open a text prompt");
    },
  },
});

const appJs = path.resolve(
  __dirname,
  "../../src/excavator_il/collection_ui_static/app.js",
);
vm.runInContext(fs.readFileSync(appJs, "utf8"), context, {filename: appJs});

const authorization = vm.runInContext(
  "hybridMotionAuthorization()",
  context,
);

assert.equal(authorization, "ALLOW_HYBRID_MACHINE_MOTION");
assert.equal(promptCalls, 0);
