import { evaluate, scoreInstrument } from "./scoring.js";
import { KEEP_SYNCED_DAYS, purgeOld, screenings, settings } from "./store.js";
import * as scanlocal from "./scanlocal.js";

// Server-less build (tools/build_site.py): no sign-in, nothing leaves the browser.
const STATIC = document.documentElement.dataset.static === "1";

const APP_VERSION = "0.1.0";
const root = document.getElementById("app");
const state = { user: settings.get("user"), token: settings.get("token"), bundle: settings.get("bundle"), draft: null, error: "", notice: "" };

// ---------------------------------------------------------------- helpers
const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const isClinician = () => state.user && state.user.role === "clinician";
const fmtDate = (iso) => (iso ? new Date(iso).toLocaleString(undefined, { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" }) : "");
const newId = () =>
  crypto.randomUUID
    ? crypto.randomUUID()
    : [...crypto.getRandomValues(new Uint8Array(16))].map((b, i) => (([4, 6, 8, 10].includes(i) ? "-" : "") + b.toString(16).padStart(2, "0"))).join("");

async function api(path, { method = "GET", body, form } = {}) {
  const headers = {};
  if (state.token) headers.Authorization = `Bearer ${state.token}`;
  if (body) headers["Content-Type"] = "application/json";
  const res = await fetch(`/api${path}`, { method, headers, body: form || (body ? JSON.stringify(body) : undefined) });
  if (res.status === 401 && path !== "/login") {
    signOut(false);
    throw new Error("Your sign-in has expired. Sign in again.");
  }
  const type = res.headers.get("Content-Type") || "";
  const data = type.includes("json") ? await res.json() : await res.blob();
  if (!res.ok && !(res.status === 422 && data.study)) throw new Error(data.error || `Request failed (${res.status}).`);
  return data;
}

async function loadConfig() {
  if (state.config) return;
  if (STATIC) {
    state.config = { public_demo: true, static: true, accounts: [] };
    return;
  }
  try {
    state.config = await (await fetch("/api/config")).json();
  } catch {
    state.config = { public_demo: false };
  }
}

async function loadBundle() {
  try {
    const res = await fetch(STATIC ? "bundle.json" : "/api/bundle");
    if (res.ok) {
      state.bundle = await res.json();
      settings.set("bundle", state.bundle);
    }
  } catch {
    /* offline: keep the copy saved on this device */
  }
}

function signOut(rerender = true) {
  state.user = null;
  state.token = null;
  settings.set("user", null);
  settings.set("token", null);
  if (rerender) location.hash = "#/login";
}

const OUTCOME_CLASS = { URGENT: "urgent", REFER: "refer", FOLLOW_UP: "follow", INCOMPLETE: "incomplete", NO_REFERRAL: "clear" };
const outcomeTitle = (o) => state.bundle.rules.outcomes[o]?.title || o;
const chip = (o) => `<span class="chip chip-${OUTCOME_CLASS[o] || "incomplete"}">${esc(outcomeTitle(o))}</span>`;

// ---------------------------------------------------------------- layout
function shell(content, { back = null, title = "" } = {}) {
  const online = navigator.onLine;
  return `
    <header class="bar">
      ${back ? `<a class="bar-back" href="${back}" aria-label="Back">Back</a>` : `<span class="bar-name">Memory Screening</span>`}
      <span class="bar-title">${esc(title)}</span>
      <span class="net ${online ? "net-on" : "net-off"}" role="status">${online ? "Online" : "Offline"}</span>
    </header>
    ${state.config && state.config.public_demo ? `<p class="demo-banner no-print" role="note">${state.config.static ? "Browser-only research demo. Nothing you enter leaves this device. Results are not diagnoses." : "Public demo. Do not enter real people's names or details. Everything here is wiped regularly, and the MRI model is a synthetic placeholder."}</p>` : ""}
    <main class="page">${content}</main>`;
}

function fail(message) {
  return `<p class="alert" role="alert">${esc(message)}</p>`;
}

// ---------------------------------------------------------------- sign in
function viewLogin() {
  return shell(`
    <h1>Sign in</h1>
    <p class="lede">Sign in once while you have signal. After that, screening works without internet for 30 days.</p>
    ${state.error ? fail(state.error) : ""}
    ${state.config && state.config.public_demo ? `<div class="panel"><h2>Demo accounts</h2><table class="scores"><tbody>${state.config.accounts.map((a) => `<tr><th scope="row">${a.role === "clinician" ? "Clinic view" : "Field worker view"}</th><td>${esc(a.username)}<br>${esc(a.password)}</td></tr>`).join("")}</tbody></table></div>` : ""}
    <form id="login" class="stack" autocomplete="on">
      <label>Username<input name="username" autocapitalize="none" autocomplete="username" required></label>
      <label>Password<input name="password" type="password" autocomplete="current-password" required></label>
      <button class="btn btn-primary" type="submit">Sign in</button>
    </form>`);
}

async function onLogin(form) {
  state.error = "";
  try {
    const fd = new FormData(form);
    const data = await api("/login", { method: "POST", body: { username: fd.get("username"), password: fd.get("password") } });
    state.user = data.user;
    state.token = data.token;
    settings.set("user", data.user);
    settings.set("token", data.token);
    await loadBundle();
    location.hash = "#/home";
  } catch (e) {
    state.error = navigator.onLine ? e.message : "You are offline. Sign in needs internet the first time.";
    render();
  }
}

// ---------------------------------------------------------------- home
async function viewHome() {
  const all = (await screenings.all()).sort((a, b) => (b.started_at || "").localeCompare(a.started_at || ""));
  const drafts = all.filter((r) => r.status === "draft");
  const waiting = all.filter((r) => r.status === "finished" && !r.synced);
  const done = all.filter((r) => r.status === "finished");
  const rejected = waiting.filter((r) => r.sync_error);

  return shell(`
    <p class="hello">${esc(state.user.name)}</p>
    <a class="btn btn-primary btn-big" href="#/new">Start a screening</a>

    ${STATIC ? `<nav class="row"><a class="btn" href="#/scan">Check a scan</a></nav>
    <section class="panel"><h2>Kept in this browser only</h2><p class="muted">This demo has no server. Screenings stay on this device, and the clinic referral queue is part of the full app, not this page.</p></section>` : ""}
    ${isClinician() && !STATIC ? `<nav class="row"><a class="btn" href="#/referrals">Referrals</a><a class="btn" href="#/scan">Check a scan</a></nav><nav class="row"><a class="btn" href="#/model">MRI model</a></nav>` : ""}

    <section class="panel ${STATIC ? "hidden" : ""}" aria-live="polite">
      <h2>${waiting.length ? `${waiting.length} screening${waiting.length > 1 ? "s" : ""} waiting to upload` : "Everything is uploaded"}</h2>
      <p class="muted">${waiting.length ? "They are saved on this device. Upload when you have signal." : `Uploaded screenings are removed from this device after ${KEEP_SYNCED_DAYS} days.`}</p>
      ${rejected.length ? fail(`${rejected.length} could not be stored: ${rejected[0].sync_error}`) : ""}
      ${state.error ? fail(state.error) : ""}
      ${waiting.length ? `<button class="btn" data-action="sync">Upload now</button>` : ""}
    </section>

    ${drafts.length ? `<h2>Unfinished</h2><ul class="list">${drafts.map((d) => `
      <li><a href="#/screen/${d.id}"><strong>${esc(d.participant.name || "No name yet")}</strong>
      <span class="muted">Started ${esc(fmtDate(d.started_at))}</span></a></li>`).join("")}</ul>` : ""}

    <h2>On this device</h2>
    ${done.length ? `<ul class="list">${done.map((d) => `
      <li><a href="#/result/${d.id}"><strong>${esc(d.participant.name)}</strong>
      <span class="muted">${esc(d.participant.code)}, ${esc(fmtDate(d.finished_at))}${d.synced ? ", uploaded" : ""}</span>
      ${chip((d.server_outcome || d.client_result.outcome))}</a></li>`).join("")}</ul>`
      : `<p class="muted">No screenings yet. Start one above.</p>`}

    ${STATIC ? "" : `<p class="foot"><button class="link" data-action="signout">Sign out</button></p>`}`);
}

async function sync() {
  if (STATIC) return;
  state.error = "";
  const waiting = (await screenings.all()).filter((r) => r.status === "finished" && !r.synced);
  if (!waiting.length) return;
  try {
    const { results } = await api("/sync", { method: "POST", body: { screenings: waiting } });
    for (const res of results) {
      const rec = waiting.find((w) => w.id === res.id);
      if (!rec) continue;
      if (res.status === "stored" || res.status === "duplicate") {
        rec.synced = true;
        rec.synced_at = new Date().toISOString();
        rec.server_outcome = res.outcome;
        rec.sync_error = null;
      } else {
        rec.sync_error = res.error || "rejected by the server";
      }
      await screenings.put(rec);
    }
  } catch (e) {
    state.error = navigator.onLine ? e.message : "No connection. Your screenings are still saved on this device.";
  }
}

// ---------------------------------------------------------------- screening wizard
function newDraft() {
  const pid = newId();
  return {
    id: newId(), status: "draft", synced: false, step: 0, started_at: new Date().toISOString(), finished_at: null,
    worker: state.user.username, app_version: APP_VERSION,
    consent: { given: false, by: "participant", at: null },
    participant: { id: pid, code: `P-${pid.slice(0, 6).toUpperCase()}`, name: "", age: null, sex: "", education_years: null, village: "", phone: "" },
    informant: { present: null, relationship: "" }, extended: false,
    answers: { csid_cog: {}, csid_inf: {}, iqcode16: {}, phq2: {} },
    red_flags: [], conditions: [], client_result: null,
  };
}

const stepsFor = (d) => ["consent", "person", "safety", "csid_cog", "informant", ...(d.informant.present && d.extended ? ["iqcode16"] : []), "phq2"];
const STEP_TITLE = { consent: "Consent", person: "About the person", safety: "Before you start", csid_cog: "Memory check", informant: "Family interview", iqcode16: "Longer family interview", phq2: "Mood check" };

function renderInstrument(id, d) {
  const inst = state.bundle.instruments[id];
  const ans = d.answers[id] || {};
  return `
    <p class="intro">${esc(inst.intro)}</p>
    ${inst.items.map((item) => {
      if (item.type === "info") return `<div class="say"><p>${esc(item.prompt)}</p><p class="muted">${esc(item.guide || "")}</p></div>`;
      const options = item.options || inst.options;
      return `
        <fieldset class="item">
          <legend>${esc(item.prompt)}</legend>
          ${item.guide ? `<p class="muted">${esc(item.guide)}</p>` : ""}
          <div class="opts opts-${options.length > 2 ? "many" : "two"}">
            ${options.map((o) => `<button type="button" class="opt" data-inst="${id}" data-item="${esc(item.id)}" data-value="${o.value}" aria-pressed="${ans[item.id] === o.value}">${esc(o.label)}</button>`).join("")}
          </div>
        </fieldset>`;
    }).join("")}
    <p class="foot"><button class="link" data-action="skip" data-inst="${id}">Could not do this section. Skip it</button></p>`;
}

function stepBody(step, d) {
  const rules = state.bundle.rules;
  const p = d.participant;
  if (step === "consent") {
    return `
      <div class="say">
        <p>"I would like to ask some questions about memory and thinking. It takes about ten minutes. It is not a diagnosis. If the answers suggest a doctor should look more closely, I will give you a referral to the clinic."</p>
        <p>"Your answers are saved on this phone and sent to the clinic team. You can stop at any time and it will not affect your care."</p>
      </div>
      <fieldset class="item"><legend>Who is giving consent?</legend>
        <div class="opts opts-two">
          <button type="button" class="opt" data-set="consent.by" data-value="participant" aria-pressed="${d.consent.by === "participant"}">The person</button>
          <button type="button" class="opt" data-set="consent.by" data-value="relative" aria-pressed="${d.consent.by === "relative"}">A relative, for them</button>
        </div>
      </fieldset>
      <label class="check"><input type="checkbox" data-bool="consent.given" ${d.consent.given ? "checked" : ""}> They understood and agreed to the screening</label>`;
  }
  if (step === "person") {
    return `
      <div class="stack">
        <label>Full name<input data-field="participant.name" value="${esc(p.name)}" autocomplete="off"></label>
        <label>Age in years<input data-field="participant.age" data-type="int" inputmode="numeric" value="${esc(p.age ?? "")}"></label>
        <fieldset class="item"><legend>Sex</legend><div class="opts opts-many">
          ${[["F", "Female"], ["M", "Male"], ["O", "Other"]].map(([v, l]) => `<button type="button" class="opt" data-set="participant.sex" data-value="${v}" aria-pressed="${p.sex === v}">${l}</button>`).join("")}
        </div></fieldset>
        <label>Years of schooling<input data-field="participant.education_years" data-type="int" inputmode="numeric" value="${esc(p.education_years ?? "")}"><span class="muted">Enter 0 if they did not go to school.</span></label>
        <label>Village or ward<input data-field="participant.village" value="${esc(p.village)}"></label>
        <label>Phone, if they have one<input data-field="participant.phone" inputmode="tel" value="${esc(p.phone)}"></label>
      </div>`;
  }
  if (step === "safety") {
    const box = (list, item) => `<label class="check"><input type="checkbox" data-list="${list}" value="${esc(item.id)}" ${d[list].includes(item.id) ? "checked" : ""}> ${esc(item.label)}</label>`;
    return `
      <h3>Does any of this apply?</h3>
      <p class="muted">Ask the person and their family. Any one of these means they need a doctor today.</p>
      ${rules.red_flags.map((f) => box("red_flags", f)).join("")}
      <h3>How are the conditions for testing?</h3>
      <p class="muted">Tick anything that could make the memory check harder than it should be.</p>
      ${rules.conditions.map((c) => box("conditions", c)).join("")}`;
  }
  if (step === "informant") {
    return `
      <fieldset class="item"><legend>Is someone here who knows the person well and sees them often?</legend>
        <div class="opts opts-two">
          <button type="button" class="opt" data-set="informant.present" data-value="true" data-rerender aria-pressed="${d.informant.present === true}">Yes</button>
          <button type="button" class="opt" data-set="informant.present" data-value="false" data-rerender aria-pressed="${d.informant.present === false}">No</button>
        </div>
      </fieldset>
      ${d.informant.present ? `
        <label>Their relationship to the person<input data-field="informant.relationship" value="${esc(d.informant.relationship)}" placeholder="Daughter, husband, neighbour"></label>
        ${renderInstrument("csid_inf", d)}
        <label class="check"><input type="checkbox" data-bool="extended" ${d.extended ? "checked" : ""}> Add the longer family interview (about 8 more minutes)</label>` : ""}
      ${d.informant.present === false ? `<p class="muted">Screening is more accurate with a family interview. If the memory check is borderline, the result will ask you to come back when a family member is present.</p>` : ""}`;
  }
  return renderInstrument(step, d);
}

function validateStep(step, d) {
  const p = d.participant;
  if (step === "consent" && !d.consent.given) return "Screening cannot start without consent.";
  if (step === "person") {
    if (!p.name.trim()) return "Enter the person's name.";
    if (!Number.isInteger(p.age) || p.age < 18 || p.age > 120) return "Enter the person's age in years.";
  }
  if (step === "informant" && d.informant.present === null) return "Choose Yes or No.";
  const inst = step === "informant" ? (d.informant.present ? "csid_inf" : null) : state.bundle.instruments[step] ? step : null;
  if (inst) {
    const s = scoreInstrument(state.bundle.instruments[inst], d.answers[inst]);
    if (s.status === "incomplete") return `${s.answered} of ${s.total} answered. Answer every question, or skip the section.`;
  }
  return "";
}

function viewScreen(d) {
  const steps = stepsFor(d);
  d.step = Math.min(d.step, steps.length - 1);
  const step = steps[d.step];
  const last = d.step === steps.length - 1;
  return shell(`
    <p class="progress-label">Step ${d.step + 1} of ${steps.length}</p>
    <progress class="progress" max="${steps.length}" value="${d.step + 1}"></progress>
    <h1>${esc(STEP_TITLE[step])}</h1>
    ${stepBody(step, d)}
    ${state.error ? fail(state.error) : ""}
    <div class="actions">
      ${d.step > 0 ? `<button class="btn" data-action="back">Back</button>` : ""}
      <button class="btn btn-primary" data-action="next">${last ? "Finish and show result" : "Next"}</button>
    </div>`, { back: "#/home", title: d.participant.name || "New screening" });
}

function setPath(obj, path, value) {
  const keys = path.split(".");
  const lastKey = keys.pop();
  keys.reduce((o, k) => o[k], obj)[lastKey] = value;
}

async function finish(d) {
  if (d.informant.present !== true) {
    d.answers.csid_inf = {};
    d.answers.iqcode16 = {};
  }
  if (!d.extended) d.answers.iqcode16 = {};
  const r = evaluate(d, state.bundle);
  d.client_result = { outcome: r.outcome, reasons: r.reasons, notes: r.notes, scores: r.scores, csid_combined: r.csid_combined, followup_months: r.followup_months, red_flags: r.red_flags, rules_version: r.rules_version };
  d.consent.at = d.consent.at || new Date().toISOString();
  d.status = "finished";
  d.finished_at = new Date().toISOString();
  await screenings.put(d);
  if (navigator.onLine) await sync();
  state.error = "";
  location.hash = `#/result/${d.id}`;
}

// ---------------------------------------------------------------- result slip
function scoreRows(scores) {
  const inst = state.bundle.instruments;
  const rows = [];
  if (scores.csid_cog != null) rows.push(["Memory check", `${scores.csid_cog} of 9`]);
  if (scores.csid_inf != null) rows.push(["Family interview", `${scores.csid_inf} of 6`]);
  if (scores.iqcode16 != null) rows.push([inst.iqcode16.short, (scores.iqcode16 / 100).toFixed(2)]);
  if (scores.phq2 != null) rows.push(["Mood check", `${scores.phq2} of 6`]);
  return rows.map(([k, v]) => `<tr><th scope="row">${esc(k)}</th><td>${esc(v)}</td></tr>`).join("");
}

function slip(participant, result, when, footer = "") {
  const rules = state.bundle.rules;
  const o = rules.outcomes[result.outcome];
  const flags = (result.red_flags || []).map((id) => rules.red_flags.find((f) => f.id === id)).filter(Boolean);
  return `
    <article class="slip slip-${OUTCOME_CLASS[result.outcome]}">
      <p class="stamp">${esc(o.title)}</p>
      <h1 class="slip-name">${esc(participant.name)}</h1>
      <p class="muted">${esc([participant.age ? `${participant.age} years` : "", participant.village, participant.code].filter(Boolean).join(", "))}</p>
      <p class="muted">Screened ${esc(fmtDate(when))}</p>

      <h2>Why</h2>
      <ul class="plain">
        ${flags.map((f) => `<li><strong>${esc(f.label)}.</strong> ${esc(f.why)}</li>`).join("")}
        ${result.reasons.filter((r) => r !== "red_flag").map((r) => `<li>${esc(rules.reasons[r] || r)}</li>`).join("") || (flags.length ? "" : "<li>Scores are in the usual range.</li>")}
        ${result.followup_months ? `<li>Check again in about ${result.followup_months} months, with a family member present.</li>` : ""}
      </ul>
      ${result.notes.length ? `<h2>Tell the clinic</h2><ul class="plain">${result.notes.map((n) => `<li>${esc(rules.notes[n] || n)}</li>`).join("")}</ul>` : ""}
      <table class="scores"><tbody>${scoreRows(result.scores)}</tbody></table>
      <p class="small">This is a screening result, not a diagnosis. A positive screen does not mean dementia, and a clear screen does not rule it out. Rules version ${esc(result.rules_version || rules.version)}.</p>
      ${footer}
    </article>`;
}

async function viewResult(id) {
  const d = await screenings.get(id);
  if (!d || !d.client_result) return shell(fail("That screening is not on this device."), { back: "#/home" });
  const o = state.bundle.rules.outcomes[d.client_result.outcome];
  const mismatch = d.server_outcome && d.server_outcome !== d.client_result.outcome;
  return shell(`
    ${slip(d.participant, d.client_result, d.finished_at)}
    <section class="panel no-print">
      <h2>What to say</h2>
      <p class="say-text">"${esc(o.say)}"</p>
    </section>
    ${mismatch ? fail(`The clinic server scored this as "${outcomeTitle(d.server_outcome)}". Follow the server result and tell your supervisor: this device may have old rules.`) : ""}
    <p class="muted no-print">${STATIC ? "Saved in this browser only." : d.synced ? "Uploaded to the clinic." : "Saved on this device. It will upload the next time you sync."}</p>
    <div class="actions no-print">
      <button class="btn" data-action="print">Print this slip</button>
      <a class="btn btn-primary" href="#/home">Done</a>
    </div>`, { back: "#/home", title: "Result" });
}

// ---------------------------------------------------------------- clinic
const STATUS_LABEL = { pending: "Waiting", seen: "Seen", scan_ordered: "Scan ordered", closed: "Closed" };

async function viewReferrals() {
  let data;
  try {
    data = await api("/referrals");
  } catch (e) {
    return shell(fail(e.message), { back: "#/home", title: "Referrals" });
  }
  const rules = state.bundle.rules;
  return shell(`
    <h1>Referrals</h1>
    ${data.referrals.length ? `<ul class="list">${data.referrals.map((r) => `
      <li><a href="#/person/${esc(r.participant.id)}">
        <strong>${esc(r.participant.name)}</strong>
        <span class="muted">${esc([r.participant.age ? `${r.participant.age} years` : "", r.participant.village].filter(Boolean).join(", "))}</span>
        <span class="muted">${esc(r.screening.reasons.map((x) => rules.reasons[x] || x).join("; "))}</span>
        <span>${chip(r.priority)} <span class="chip">${esc(STATUS_LABEL[r.status])}</span></span>
      </a></li>`).join("")}</ul>` : `<p class="muted">No referrals yet. They appear here when field workers upload a screening that needs assessment.</p>`}`,
    { back: "#/home", title: "Clinic" });
}

function mriCard(m, model) {
  if (m.status === "rejected") {
    return `<div class="panel"><h3>${esc(m.filename)}</h3><p class="muted">${esc(fmtDate(m.created_at))}</p>${fail(m.result.reason)}</div>`;
  }
  const r = m.result;
  const pct = (v) => `${Math.round(v * 100)}%`;
  return `
    <div class="panel">
      <h3>${esc(m.filename)}</h3>
      <p class="muted">${esc(fmtDate(m.created_at))}, model ${esc(r.model)}</p>
      ${r.synthetic ? fail("Demo model trained on synthetic noise. This output is meaningless and is here only to show how the page works.") : ""}
      <p class="mri-score">${pct(r.probability)} <span class="muted">model score, threshold ${pct(r.threshold)}</span></p>
      <meter class="meter" min="0" max="1" value="${r.probability}" aria-label="Model score"></meter>
      <p>${r.above_threshold ? "Above the threshold: the scan resembles the dementia group in the training data." : "Below the threshold: the scan resembles the group without dementia in the training data."}</p>
      <p class="muted">${r.member_probabilities.length} models scored it ${r.member_probabilities.map(pct).join(", ")}.
        ${r.members_agree ? "They agree on which side of the threshold it falls." : "<strong>They disagree about which side of the threshold it falls. Treat this score as uninformative.</strong>"}</p>
      <figure class="mri-fig" data-study="${m.id}"><img alt="Three views of the scan as the model received it" data-mri="${m.id}"><figcaption class="muted">The scan as the model received it, at 2 mm.</figcaption></figure>
      <button class="btn" data-action="explain" data-study="${m.id}">Show which regions moved the score (1 to 2 minutes)</button>
      <p class="small">Decision support for research use. It cannot tell what is causing a person's symptoms, and it has only been evaluated on ${esc(model?.training?.dataset || "its training cohort")}. The diagnosis is the clinician's.</p>
    </div>`;
}

async function viewPerson(pid) {
  let data, model;
  try {
    [data, model] = await Promise.all([api(`/participants/${pid}`), api("/model")]);
  } catch (e) {
    return shell(fail(e.message), { back: "#/referrals", title: "Person" });
  }
  const latest = data.screenings[0];
  const ref = data.referrals[0];
  return shell(`
    ${latest ? slip(data.participant, { ...latest.result, rules_version: latest.result.rules_version }, latest.finished_at,
      `<p class="small">Screened by ${esc(latest.worker || "")}.${latest.informant?.present ? ` Family member: ${esc(latest.informant.relationship || "not recorded")}.` : " No family interview."}</p>`) : ""}
    ${ref ? `
      <section class="panel"><h2>Referral</h2>
        <form id="referral" data-ref="${ref.id}" class="stack">
          <label>Status<select name="status">${Object.entries(STATUS_LABEL).map(([v, l]) => `<option value="${v}" ${ref.status === v ? "selected" : ""}>${l}</option>`).join("")}</select></label>
          <label>Clinic notes<textarea name="notes" rows="3">${esc(ref.notes)}</textarea></label>
          <button class="btn" type="submit">Save referral</button>
          ${state.notice ? `<p class="notice" role="status">${esc(state.notice)}</p>` : ""}
        </form></section>` : ""}
    <section>
      <h2>MRI analysis</h2>
      ${state.error ? fail(state.error) : ""}
      ${model.loaded ? `
        <form id="mri" data-pid="${esc(pid)}" class="panel stack">
          <label>Scan file<input type="file" name="files" multiple required accept=".nii,.gz,.img,.hdr"></label>
          <p class="muted">One NIfTI file, or an Analyze .hdr and .img pair. The scan must be skull-stripped and registered to the T88 atlas at 1 mm, like OASIS-1. Raw scanner output is refused.</p>
          <button class="btn btn-primary" type="submit">Analyse scan</button>
        </form>` : `<p class="panel muted">${esc(model.message)}</p>`}
      ${data.mri.map((m) => mriCard(m, model)).join("")}
    </section>`, { back: "#/referrals", title: data.participant.name });
}

async function loadMriImages(overlayFor = null) {
  for (const img of root.querySelectorAll("img[data-mri]")) {
    const id = img.dataset.mri;
    try {
      const blob = await api(`/mri/${id}/image${overlayFor === id ? "?overlay=1" : ""}`);
      img.src = URL.createObjectURL(blob);
    } catch {
      img.alt = "The image could not be loaded.";
    }
  }
}

// ---------------------------------------------------------------- scan check
const BAND = {
  usual: { title: "Within the usual range for age", cls: "clear" },
  lower: { title: "Lower than usual for age", cls: "follow" },
  well_below: { title: "Well below the usual range for age", cls: "refer" },
};
const BAND_PLAIN = {
  usual: "The amount of brain tissue measured on this scan is in the range usually seen in people of the same age who do not have dementia.",
  lower: "The amount of brain tissue measured on this scan is somewhat lower than is usual for people of the same age who do not have dementia.",
  well_below: "The amount of brain tissue measured on this scan is clearly lower than is usual for people of the same age who do not have dementia.",
};

function scanResult(r) {
  if (!r.accepted) return `<section class="panel"><h2>Scan not accepted</h2>${fail(r.reason)}</section>`;
  const sampleNote = r.sample ? fail("Synthetic sample, not a real brain. These numbers are meaningless and only show the layout.") : "";
  const m = r.measure, v = r.reference.validation, b = BAND[m.band];
  const pct = (x) => `${Math.round(x * 100)}%`;
  const model = r.model;
  return `
    ${sampleNote}
    <div class="tabs no-print" role="tablist">
      <button class="opt" data-tab="doctor" aria-pressed="${state.scanTab !== "patient"}">For the doctor</button>
      <button class="opt" data-tab="patient" aria-pressed="${state.scanTab === "patient"}">For the patient</button>
    </div>
    ${state.scanTab !== "patient" ? `
    <article class="slip slip-${b.cls}">
      <p class="stamp">${esc(b.title)}</p>
      <table class="scores"><tbody>
        <tr><th scope="row">Brain tissue fraction</th><td>${m.brain_fraction.toFixed(3)}</td></tr>
        <tr><th scope="row">Expected at age ${esc(m.age)}</th><td>${m.expected_for_age.toFixed(3)}</td></tr>
        <tr><th scope="row">z-score against reference</th><td>${m.z.toFixed(2)}</td></tr>
        <tr><th scope="row">Percentile among people without dementia</th><td>${m.percentile.toFixed(0)}</td></tr>
      </tbody></table>
      ${m.age_in_reference_range ? "" : fail("This age is outside the reference range, so the comparison is extrapolated.")}
      <figure class="mri-fig"><img alt="Three views of the scan as analysed" src="${r.image}"><figcaption class="muted">The scan as analysed.</figcaption></figure>
      <h2>How far to trust this number</h2>
      <ul class="plain">
        <li>Repeatable: ICC ${v.test_retest_icc} across ${v.test_retest_pairs} people scanned twice.</li>
        <li>Agrees with OASIS's own brain-volume measure: r = ${v.corr_with_oasis_nwbv}.</li>
        <li>Weak at separating groups: with age, AUC ${v.auc_with_age_cdr0_vs_cdr_ge_0_5} (95% CI ${v.auc_ci95[0]} to ${v.auc_ci95[1]}) for dementia against no dementia. The two groups overlap heavily.</li>
        <li>Very mild dementia (CDR 0.5) was mostly missed: in the same study the age-and-volume model detected about ${pct(v.detected_cdr_0_5_at_80pct_specificity_tabular_model)} at 80% specificity, and the scan-based models did no better.</li>
      </ul>
      <h2>Limits</h2>
      <ul class="plain">${r.reference.limits.map((l) => `<li>${esc(l)}</li>`).join("")}</ul>
      ${model ? `
        <h2>Neural network score</h2>
        ${model.synthetic ? fail("Demo model trained on synthetic noise. This score is meaningless.") : ""}
        <p>${pct(model.probability)} (threshold ${pct(model.threshold)}; ${model.member_probabilities.length} models: ${model.member_probabilities.map(pct).join(", ")}).
        ${model.evaluation && typeof model.evaluation.auc === "number" ? `In testing this network reached AUC ${model.evaluation.auc.toFixed(2)}, below the brain-volume measure above. Do not weigh it.` : ""}</p>` : `<p class="muted">${esc(r.model_message || "")}</p>`}
      <p class="small">Decision support for research use. Not a diagnosis and not a medical device. Reference: ${esc(r.reference.reference_group)}.</p>
    </article>` : `
    <article class="slip slip-${b.cls}">
      <h1 class="slip-name">About your brain scan</h1>
      <h2>What was measured</h2>
      <p>A computer measured how much of the space inside the skull is filled by brain tissue. Everyone's brain gets a little smaller with age, so the result is compared with people of the same age.</p>
      <h2>What it found</h2>
      <p>${BAND_PLAIN[m.band]}</p>
      <h2>What this does not tell you</h2>
      <ul class="plain">
        <li>It does not say whether you have Alzheimer's disease or any other illness. A scan measurement cannot do that by itself.</li>
        <li>Many people with a lower measurement have no memory illness, and some people with a memory illness have a usual measurement.</li>
        <li>It says nothing about a stage or about what will happen in future.</li>
      </ul>
      <h2>What happens next</h2>
      <p>Your doctor puts this together with your symptoms, memory tests, blood tests and examination. Only your doctor can tell you what it means for you.</p>
      <h2>Questions you may want to ask</h2>
      <ul class="plain">
        <li>What do you think is causing my symptoms?</li>
        <li>Are there other tests I should have?</li>
        <li>Is there anything treatable that could be affecting my memory?</li>
        <li>What should my family and I watch for?</li>
      </ul>
      <p class="small">This summary was produced by research software. It is not a diagnosis.</p>
    </article>`}
    <div class="actions no-print"><button class="btn" data-action="print">Print this page</button></div>`;
}

async function scanInBrowser(form, sample) {
  const age = Number(form.querySelector("input[name=age]").value);
  if (!(age >= 18 && age <= 110)) throw new Error("Enter an age between 18 and 110.");
  try {
    const ref = await (await fetch("reference.json")).json();
    const vol = sample ? scanlocal.sampleVolume() : await scanlocal.readFiles([...form.querySelector("input[type=file]").files]);
    const checks = scanlocal.checkInput(vol);
    const { measure, scale } = scanlocal.assess(vol.data, age, ref);
    return { accepted: true, sample, checks, measure, image: scanlocal.renderSlices(vol, scale),
      reference: { name: ref.name, validation: ref.validation, limits: ref.limits, reference_group: ref.reference_group },
      model_message: "The browser-only demo does not include the neural network. In testing a single trained network scored lower than this measure, so nothing is lost." };
  } catch (e) {
    if (e instanceof scanlocal.ScanRejected) return { accepted: false, reason: e.message };
    throw e;
  }
}

function viewScan() {
  return shell(`
    <h1>Check a scan</h1>
    <p class="lede">Measures brain tissue on one scan and compares it with people of the same age. ${STATIC ? "The scan is analysed in your browser and never uploaded." : "Nothing is saved."}</p>
    <form id="scan" class="panel stack no-print">
      <label>Scan file<input type="file" name="files" multiple accept=".nii,.gz,.img,.hdr"></label>
      <p class="muted">One NIfTI file, or an Analyze .hdr and .img pair, prepared like OASIS-1: skull-stripped and registered to the T88 atlas at 1 mm.</p>
      <label>Age in years<input name="age" inputmode="numeric" required></label>
      <button class="btn btn-primary" type="submit">Analyse scan</button>
      <button class="btn" type="submit" name="sample" value="1">Try a synthetic sample instead</button>
      <p class="muted">The sample is a computer-generated shape, not a real brain. It only shows how the page works.</p>
    </form>
    ${state.error ? fail(state.error) : ""}
    ${state.scan ? scanResult(state.scan) : ""}`, { back: "#/home", title: "Clinic" });
}

async function viewModel() {
  let m;
  try {
    m = await api("/model");
  } catch (e) {
    return shell(fail(e.message), { back: "#/home", title: "MRI model" });
  }
  if (!m.loaded) return shell(`<h1>MRI model</h1><p class="panel">${esc(m.message)}</p>`, { back: "#/home", title: "Clinic" });
  const e = m.evaluation || {};
  const num = (v, d = 3) => (typeof v === "number" ? v.toFixed(d) : "not available");
  return shell(`
    <h1>${esc(m.name)}</h1>
    ${m.synthetic ? fail("Demo model trained on synthetic noise. Its scores are meaningless.") : ""}
    <p class="lede">${esc(m.task)}</p>
    <table class="scores"><tbody>
      <tr><th scope="row">Trained on</th><td>${esc(m.training?.dataset)}, ${esc(m.training?.n_subjects)} people (${esc(m.training?.n_positive)} with dementia)</td></tr>
      <tr><th scope="row">How it was tested</th><td>${esc(e.method)}</td></tr>
      <tr><th scope="row">AUC</th><td>${num(e.auc)}${e.auc_ci95 ? ` (95% CI ${num(e.auc_ci95[0])} to ${num(e.auc_ci95[1])})` : ""}</td></tr>
      <tr><th scope="row">Sensitivity, specificity</th><td>${num(e.sensitivity)}, ${num(e.specificity)}</td></tr>
      <tr><th scope="row">Brain volume + age baseline AUC</th><td>${num(e.baseline_tabular_auc)}</td></tr>
      <tr><th scope="row">Age only AUC</th><td>${num(e.baseline_age_auc)}</td></tr>
      <tr><th scope="row">Models in the ensemble</th><td>${esc(m.members)}</td></tr>
    </tbody></table>
    <h2>Limits</h2>
    <ul class="plain">${m.limits.map((l) => `<li>${esc(l)}</li>`).join("")}</ul>`, { back: "#/home", title: "Clinic" });
}

// ---------------------------------------------------------------- router
async function render() {
  const hash = location.hash || "#/home";
  const [, route, arg] = hash.split("/");
  if (!state.bundle) await loadBundle();
  await loadConfig();
  if (STATIC && !state.user) {
    state.user = { username: "visitor", name: "Research demo", role: "clinician" };
    state.token = "static";
  }
  if (!state.user || !state.token) {
    root.innerHTML = viewLogin();
    return;
  }
  if (!state.bundle) {
    root.innerHTML = shell(fail("The screening questions could not be loaded. Connect to the internet once and reopen the app."));
    return;
  }
  let html;
  if (route === "new") {
    const d = newDraft();
    await screenings.put(d);
    location.replace(`#/screen/${d.id}`);
    return;
  } else if (route === "screen") {
    state.draft = await screenings.get(arg);
    if (!state.draft || state.draft.status !== "draft") {
      location.replace(state.draft ? `#/result/${arg}` : "#/home");
      return;
    }
    html = viewScreen(state.draft);
  } else if (route === "result") html = await viewResult(arg);
  else if (route === "referrals" && isClinician() && !STATIC) html = await viewReferrals();
  else if (route === "person" && isClinician() && !STATIC) html = await viewPerson(arg);
  else if (route === "model" && isClinician() && !STATIC) html = await viewModel();
  else if (route === "scan" && isClinician()) html = viewScan();
  else if (route === "login") {
    location.replace("#/home");
    return;
  } else html = await viewHome();
  root.innerHTML = html;
  if (route === "person") loadMriImages();
}

// ---------------------------------------------------------------- events
root.addEventListener("click", async (ev) => {
  const el = ev.target.closest("button, a");
  if (!el || el.tagName === "A") return;
  const d = state.draft;

  if (el.dataset.tab) {
    state.scanTab = el.dataset.tab;
    return render();
  }
  if (el.classList.contains("opt")) {
    for (const sib of el.parentElement.querySelectorAll(".opt")) sib.setAttribute("aria-pressed", String(sib === el));
    if (el.dataset.inst) d.answers[el.dataset.inst][el.dataset.item] = Number(el.dataset.value);
    else if (el.dataset.set) {
      const raw = el.dataset.value;
      setPath(d, el.dataset.set, raw === "true" ? true : raw === "false" ? false : raw);
    }
    await screenings.put(d);
    if ("rerender" in el.dataset) render();
    return;
  }

  const action = el.dataset.action;
  if (action === "next" || action === "back") {
    const steps = stepsFor(d);
    if (action === "back") {
      state.error = "";
      d.step = Math.max(0, d.step - 1);
    } else {
      state.error = validateStep(steps[d.step], d);
      if (!state.error) {
        if (d.step === steps.length - 1) return finish(d);
        d.step += 1;
      }
    }
    await screenings.put(d);
    await render();
    if (!state.error) window.scrollTo(0, 0);
  } else if (action === "skip") {
    d.answers[el.dataset.inst] = {};
    await screenings.put(d);
    state.error = "";
    render();
  } else if (action === "sync") {
    el.disabled = true;
    el.textContent = "Uploading";
    await sync();
    render();
  } else if (action === "signout") {
    const waiting = (await screenings.all()).filter((r) => r.status === "finished" && !r.synced).length;
    if (waiting) {
      state.error = `Upload the ${waiting} waiting screening${waiting > 1 ? "s" : ""} before signing out.`;
      render();
    } else signOut();
  } else if (action === "print") {
    window.print();
  } else if (action === "explain") {
    el.disabled = true;
    el.textContent = "Working. This takes a minute or two";
    try {
      await api(`/mri/${el.dataset.study}/explain`, { method: "POST" });
      await loadMriImages(el.dataset.study);
      el.textContent = "Orange raised the score, blue lowered it. Blocks are 4 cm wide";
    } catch (e) {
      el.textContent = e.message;
    }
  }
});

root.addEventListener("input", async (ev) => {
  const el = ev.target;
  const d = state.draft;
  if (!d) return;
  if (el.dataset.field) {
    let value = el.value;
    if (el.dataset.type === "int") value = /^\d+$/.test(value.trim()) ? Number(value.trim()) : null;
    setPath(d, el.dataset.field, value);
  } else if (el.dataset.bool) {
    setPath(d, el.dataset.bool, el.checked);
    if (el.dataset.bool === "consent.given") d.consent.at = el.checked ? new Date().toISOString() : null;
  } else if (el.dataset.list) {
    const list = d[el.dataset.list];
    if (el.checked && !list.includes(el.value)) list.push(el.value);
    if (!el.checked) d[el.dataset.list] = list.filter((v) => v !== el.value);
  } else return;
  await screenings.put(d);
});

root.addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const form = ev.target;
  if (form.id === "login") return onLogin(form);
  state.error = "";
  state.notice = "";
  const button = (ev.submitter && ev.submitter.type === "submit") ? ev.submitter : form.querySelector("button[type=submit]");
  button.disabled = true;
  try {
    if (form.id === "referral") {
      const fd = new FormData(form);
      await api(`/referrals/${form.dataset.ref}`, { method: "PATCH", body: { status: fd.get("status"), notes: fd.get("notes") } });
      state.notice = "Referral saved.";
    } else if (form.id === "scan") {
      button.textContent = "Analysing";
      state.scan = null;
      const fd = new FormData(form);
      if (ev.submitter && ev.submitter.name === "sample") fd.set("sample", "1");
      else if (!form.querySelector("input[type=file]").files.length) throw new Error("Choose a scan file, or try the synthetic sample.");
      if (STATIC) {
        state.scan = await scanInBrowser(form, fd.get("sample") === "1");
      } else {
        const res = await fetch("/api/scan", { method: "POST", headers: { Authorization: `Bearer ${state.token}` }, body: fd });
        const data = await res.json();
        if (!res.ok && res.status !== 422) throw new Error(data.error || `Request failed (${res.status}).`);
        state.scan = data;
      }
    } else if (form.id === "mri") {
      button.textContent = "Analysing";
      await api(`/participants/${form.dataset.pid}/mri`, { method: "POST", form: new FormData(form) });
    }
  } catch (e) {
    state.error = e.message;
  }
  render();
});

window.addEventListener("hashchange", () => {
  state.error = "";
  state.notice = "";
  render();
});
window.addEventListener("online", async () => {
  if (state.token) await sync();
  render();
});
window.addEventListener("offline", render);

if ("serviceWorker" in navigator) navigator.serviceWorker.register(STATIC ? "sw.js" : "/sw.js").catch(() => {});
purgeOld().catch(() => {});
render();
