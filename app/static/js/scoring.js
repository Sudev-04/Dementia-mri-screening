// Offline scoring. This is a line-for-line port of common/scoring.py.
// tests/js/scoring_parity.mjs checks both against instruments/test_vectors.json,
// so change the two files together.

export const OUTCOME_RANK = { NO_REFERRAL: 0, INCOMPLETE: 1, FOLLOW_UP: 2, REFER: 3, URGENT: 4 };

function allowedValues(instrument, item) {
  const options = item.options || instrument.options || [];
  return new Set(options.map((o) => o.value));
}

export function scoreInstrument(instrument, answers) {
  answers = answers || {};
  const scored = instrument.items.filter((it) => it.type !== "info");
  const values = [];
  for (const item of scored) {
    const raw = answers[item.id];
    if (typeof raw !== "number" || !Number.isInteger(raw)) continue;
    if (allowedValues(instrument, item).has(raw)) values.push(raw);
  }
  const total = scored.length;
  const answered = values.length;
  const result = { status: "not_done", score: null, answered, total };
  if (answered === 0) return result;

  const scoring = instrument.scoring;
  if (total - answered > (scoring.max_missing || 0)) {
    result.status = "incomplete";
    return result;
  }
  const s = values.reduce((a, b) => a + b, 0);
  result.score = scoring.method === "mean" ? Math.floor((200 * s + answered) / (2 * answered)) : s;
  result.status = "complete";
  return result;
}

export function decide(scores, redFlags, conditions, rules) {
  const t = {};
  for (const [k, v] of Object.entries(rules.thresholds)) t[k] = v.value;
  const flags = rules.red_flags.map((f) => f.id).filter((id) => (redFlags || []).includes(id));
  const conds = rules.conditions.map((c) => c.id).filter((id) => (conditions || []).includes(id));

  const cog = scores.csid_cog ?? null;
  const inf = scores.csid_inf ?? null;
  const iq = scores.iqcode16 ?? null;
  const phq = scores.phq2 ?? null;

  let outcome = "NO_REFERRAL";
  const reasons = [];
  const notes = [];
  let combined = null;
  let cogDroveResult = false;

  const raiseTo = (level, reason) => {
    if (OUTCOME_RANK[level] > OUTCOME_RANK[outcome]) outcome = level;
    reasons.push(reason);
  };

  if (flags.length) raiseTo("URGENT", "red_flag");

  if (cog !== null && inf !== null) {
    combined = cog - inf;
    if (combined <= t.csid_combined_max) {
      raiseTo("REFER", "csid_combined_positive");
      cogDroveResult = true;
    } else if (inf >= t.csid_inf_min) {
      raiseTo("FOLLOW_UP", "informant_concern");
    }
  } else if (cog !== null) {
    if (cog <= t.csid_cog_positive_max) {
      raiseTo("REFER", "csid_cog_positive");
      cogDroveResult = true;
    } else if (cog <= t.csid_cog_indeterminate_max) {
      raiseTo("FOLLOW_UP", "csid_cog_indeterminate");
      cogDroveResult = true;
    }
  } else if (inf !== null) {
    if (inf >= t.csid_inf_min) raiseTo("REFER", "csid_inf_positive");
  }

  if (iq !== null && iq >= t.iqcode16_min_x100) raiseTo("REFER", "iqcode_positive");
  if (cog === null && inf === null && iq === null) raiseTo("INCOMPLETE", "nothing_scored");

  if (phq !== null && phq >= t.phq2_min) notes.push("phq2_positive");
  if (conds.length && cogDroveResult) notes.push("test_conditions");
  if (cog !== null && inf === null && iq === null) notes.push("no_informant");

  return {
    outcome,
    reasons,
    notes,
    red_flags: flags,
    conditions: conds,
    csid_combined: combined,
    followup_months: outcome === "FOLLOW_UP" ? t.followup_months : null,
  };
}

export function evaluate(record, bundle) {
  const answers = record.answers || {};
  const detail = {};
  const scores = {};
  for (const [id, inst] of Object.entries(bundle.instruments)) {
    detail[id] = scoreInstrument(inst, answers[id]);
    scores[id] = detail[id].score;
  }
  const decision = decide(scores, record.red_flags, record.conditions, bundle.rules);
  return { rules_version: bundle.rules.version, scores, detail, ...decision };
}
