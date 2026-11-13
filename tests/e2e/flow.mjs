// Browser test of the whole journey: field screening online, field screening
// with no network, automatic upload, then the clinic's referral and MRI pages.
//
//   python run_app.py seed-demo
//   python run_app.py --demo-model --port 5055
//   node tests/e2e/flow.mjs http://127.0.0.1:5055 path/to/scan.hdr path/to/scan.img [screenshot-dir]
//
// Needs Playwright (npm i -D playwright). Not part of the pytest run.
import { chromium, devices } from "playwright";

const [base = "http://127.0.0.1:5055", hdr, img, shots] = process.argv.slice(2);
const assert = (cond, msg) => { if (!cond) throw new Error(`FAILED: ${msg}`); console.log(`ok  ${msg}`); };
const snap = async (page, name) => { if (shots) await page.screenshot({ path: `${shots}/${name}.png`, fullPage: true }); };

async function signIn(page, username, password) {
  await page.goto(base);
  await page.fill("input[name=username]", username);
  await page.fill("input[name=password]", password);
  await page.click("button[type=submit]");
  await page.waitForSelector(".hello");
}

async function answer(page, itemId, value) {
  await page.click(`.opt[data-item="${itemId}"][data-value="${value}"]`);
}
const next = (page) => page.click('[data-action="next"]');

async function screen(page, { name, cog, informant }) {
  await page.click('a[href="#/new"]');
  await page.waitForSelector("text=Consent");
  await next(page);
  assert(await page.locator(".alert").innerText() === "Screening cannot start without consent.", "consent is required to continue");
  await page.check('input[data-bool="consent.given"]');
  await next(page);
  await page.fill('input[data-field="participant.name"]', name);
  await page.fill('input[data-field="participant.age"]', "74");
  await page.click('.opt[data-set="participant.sex"][data-value="F"]');
  await page.fill('input[data-field="participant.education_years"]', "3");
  await page.fill('input[data-field="participant.village"]', "Ward 7");
  await next(page);
  await page.check('input[data-list="conditions"][value="hearing"]');
  await next(page);
  for (const [item, value] of Object.entries(cog)) await answer(page, item, value);
  await next(page);
  if (informant) {
    await page.click('.opt[data-set="informant.present"][data-value="true"]');
    await page.fill('input[data-field="informant.relationship"]', "Daughter");
    for (const [item, value] of Object.entries(informant)) await answer(page, item, value);
  } else {
    await page.click('.opt[data-set="informant.present"][data-value="false"]');
  }
  await next(page);
  await answer(page, "interest", 0);
  await answer(page, "mood", 1);
  await next(page);
  await page.waitForSelector(".slip");
}

const browser = await chromium.launch();
const errors = [];
try {
  // ---------------------------------------------------------------- field worker on a phone
  const phone = await browser.newContext({ ...devices["Pixel 7"] });
  const page = await phone.newPage();
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("console", (m) => { if (m.type() === "error" && !/ERR_INTERNET_DISCONNECTED|Failed to load resource/.test(m.text())) errors.push(m.text()); });

  await page.goto(base);
  await snap(page, "01-sign-in");
  await signIn(page, "field1", "field-demo-1");
  await page.evaluate(() => navigator.serviceWorker.ready);
  await snap(page, "02-home");

  const low = { elbow: 1, hammer: 1, store: 0, day: 0, season: 0, point: 1, recall: 0 };
  const informant = { decline: 1, reason: 1, misplace: 1, yesterday: 1, where: 0, dressing: 0 };
  await page.click('a[href="#/new"]');
  await page.waitForSelector("text=Consent");
  await snap(page, "03-consent");
  await page.goto(`${base}/#/home`);
  await page.waitForSelector(".hello");

  await screen(page, { name: "Online Person", cog: low, informant });
  await snap(page, "05-result-refer");
  assert((await page.locator(".stamp").innerText()).includes("Refer to the clinic"), "low score with informant concern gives a referral");
  assert((await page.locator(".slip").innerText()).includes("3 of 9"), "slip shows the memory check score");
  assert(await page.locator("text=Uploaded to the clinic.").count() === 1, "online screening uploads straight away");

  // ---- no signal: reload must still work, and a screening must be saved locally
  await phone.setOffline(true);
  await page.goto(`${base}/#/home`);
  await page.reload();
  await page.waitForSelector(".hello");
  assert(await page.locator(".net-off").count() === 1, "app opens with no network and says it is offline");

  const high = { elbow: 1, hammer: 1, store: 1, day: 1, season: 1, point: 1, recall: 2 };
  await page.click('a[href="#/new"]');
  await page.waitForSelector("text=Consent");
  await page.goto(`${base}/#/home`);
  await page.waitForSelector(".hello");
  await screen(page, { name: "Offline Person", cog: { ...high, recall: 0 }, informant: null });
  await snap(page, "06-result-offline");
  assert((await page.locator(".stamp").innerText()).includes("Check again"), "borderline score without informant asks for a re-check");
  assert(await page.locator("text=Saved on this device").count() === 1, "offline screening is kept on the device");
  await page.click('a.btn-primary[href="#/home"]');
  await page.waitForSelector("text=waiting to upload");
  await snap(page, "07-home-offline");

  // ---- signal returns: upload happens without the worker doing anything
  await phone.setOffline(false);
  await page.waitForSelector("text=Everything is uploaded", { timeout: 15000 });
  assert(true, "waiting screenings upload automatically when the network returns");
  assert(await page.locator('a[href="#/referrals"]').count() === 0, "field workers are not shown clinic pages");
  await phone.close();

  // ---------------------------------------------------------------- clinician on a laptop
  const desk = await browser.newContext({ viewport: { width: 1100, height: 900 } });
  const cp = await desk.newPage();
  cp.on("pageerror", (e) => errors.push(e.message));
  await signIn(cp, "clinic1", "clinic-demo-1");
  await cp.click('a[href="#/referrals"]');
  await cp.waitForSelector("text=Online Person");
  await snap(cp, "08-referrals");
  assert(await cp.locator("text=Offline Person").count() === 0, "a re-check is not sent to the clinic as a referral");
  await cp.click("text=Online Person");
  await cp.waitForSelector("#mri");
  assert((await cp.locator(".slip").innerText()).includes("Testing conditions were not ideal"), "clinic sees the testing-conditions note");

  await cp.selectOption("#referral select", "scan_ordered");
  await cp.fill("#referral textarea", "MRI booked for Friday");
  await cp.click("#referral button[type=submit]");
  await cp.waitForSelector("text=Referral saved.");
  assert(await cp.locator("#referral select").inputValue() === "scan_ordered", "referral status is saved");

  if (hdr && img) {
    await cp.setInputFiles("#mri input[type=file]", [hdr, img]);
    await cp.click("#mri button[type=submit]");
    await cp.waitForSelector(".mri-score", { timeout: 60000 });
    await cp.waitForFunction(() => document.querySelector("img[data-mri]")?.naturalWidth > 0, null, { timeout: 30000 });
    assert((await cp.locator(".alert").first().innerText()).includes("Demo model"), "demo model output carries a warning");
    await snap(cp, "09-mri-result");
    await cp.click('[data-action="explain"]');
    await cp.waitForSelector("text=Orange raised the score", { timeout: 300000 });
    await cp.waitForTimeout(500);
    await snap(cp, "10-mri-explained");
    assert(true, "region map is produced");
  }
  await cp.goto(`${base}/#/model`);
  await cp.waitForSelector("text=Limits");
  await snap(cp, "11-model-card");
  await desk.close();

  assert(errors.length === 0, `no browser errors (${errors.join(" | ") || "none"})`);
  console.log("\nAll browser checks passed.");
} finally {
  await browser.close();
}
