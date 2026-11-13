// Checks the in-browser scan measurement against values computed by the Python code.
// Usage: node tests/js/scan_parity.mjs scan.hdr scan.img reference.json age expected.json
import { readFileSync } from "node:fs";
import { assess, checkInput, parseVolume } from "../../app/static/js/scanlocal.js";

const [hdr, img, refPath, age, expectedPath] = process.argv.slice(2);
const buf = (p) => { const b = readFileSync(p); return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength); };
const vol = parseVolume(buf(hdr), buf(img));
const checks = checkInput(vol);
const { measure } = assess(vol.data, Number(age), JSON.parse(readFileSync(refPath, "utf8")));
const want = JSON.parse(readFileSync(expectedPath, "utf8"));
const close = (a, b, tol) => Math.abs(a - b) <= tol;
const ok = close(measure.brain_fraction, want.brain_fraction, 2e-4) && close(measure.z, want.z, 0.02) && close(measure.percentile, want.percentile, 0.2)
  && measure.band === want.band && close(checks.background_fraction, want.background_fraction, 1e-4);
console.log(JSON.stringify({ js: { ...measure, ...checks }, python: want, ok }));
process.exit(ok ? 0 : 1);
