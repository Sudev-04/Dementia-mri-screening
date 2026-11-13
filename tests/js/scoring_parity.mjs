// Checks the browser scoring code against the vectors produced by the Python
// implementation. Run with:  node tests/js/scoring_parity.mjs
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { decide, scoreInstrument } from "../../app/static/js/scoring.js";

const root = join(dirname(fileURLToPath(import.meta.url)), "..", "..");
const load = (name) => JSON.parse(readFileSync(join(root, "instruments", name), "utf8"));
const instruments = Object.fromEntries(["csid_cog", "csid_inf", "iqcode16", "phq2"].map((id) => [id, load(`${id}.json`)]));
const rules = load("rules.json");
const vectors = load("test_vectors.json");

let failures = 0;
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);

for (const c of vectors.scoring) {
  const got = scoreInstrument(instruments[c.instrument], c.answers);
  if (!same(got, c.expected)) {
    failures += 1;
    console.error("scoring mismatch", c.instrument, JSON.stringify(c.answers), got, c.expected);
  }
}
for (const [cog, inf, iq, phq, flag, cond, outcome, reasons, notes, combined] of vectors.decide) {
  const got = decide({ csid_cog: cog, csid_inf: inf, iqcode16: iq, phq2: phq }, flag ? ["stroke_signs"] : [], cond ? ["hearing"] : [], rules);
  if (got.outcome !== outcome || got.reasons.join(",") !== reasons || got.notes.join(",") !== notes || got.csid_combined !== combined) {
    failures += 1;
    if (failures < 10) console.error("decision mismatch", { cog, inf, iq, phq, flag, cond }, got, { outcome, reasons, notes, combined });
  }
}
if (vectors.rules_version !== rules.version) {
  failures += 1;
  console.error("test_vectors.json was generated for another rules version; run: python -m tests.make_vectors");
}
console.log(`${vectors.scoring.length} scoring cases, ${vectors.decide.length} decision cases, ${failures} failures`);
process.exit(failures ? 1 : 0);
