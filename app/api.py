"""REST API."""
from __future__ import annotations

import re
import uuid
from functools import wraps
from pathlib import Path

import numpy as np
from flask import Blueprint, Response, current_app, g, jsonify, request
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from sqlalchemy import func

from common.measure import assess, load_reference
from common.preprocess import InputRejected, PreprocessConfig, check_input, load_volume, preprocess
from common.scoring import evaluate, load_bundle

from .models import MriStudy, Participant, Referral, Screening, User, db
from .mri_service import load_model, render_slices

api = Blueprint("api", __name__)
ID_RE = re.compile(r"^[A-Za-z0-9-]{8,64}$")
MRI_SUFFIXES = (".nii", ".nii.gz", ".img", ".hdr")
_model_cache: dict = {}


def err(message: str, status: int = 400):
    return jsonify(error=message), status


def _serializer() -> URLSafeTimedSerializer:
    return URLSafeTimedSerializer(current_app.config["SECRET_KEY"], salt="auth")


def require(*roles):
    def deco(fn):
        @wraps(fn)
        def wrapper(*a, **kw):
            header = request.headers.get("Authorization", "")
            if not header.startswith("Bearer "):
                return err("Sign in to continue.", 401)
            try:
                data = _serializer().loads(header[7:], max_age=current_app.config["TOKEN_MAX_AGE"])
            except SignatureExpired:
                return err("Your sign-in has expired. Sign in again.", 401)
            except BadSignature:
                return err("Sign in to continue.", 401)
            user = db.session.get(User, data.get("uid"))
            if user is None:
                return err("Sign in to continue.", 401)
            if roles and user.role not in roles:
                return err("Your account cannot open this.", 403)
            g.user = user
            return fn(*a, **kw)
        return wrapper
    return deco


def get_model():
    key = (current_app.config["MODEL_DIR"], current_app.config.get("ALLOW_DEMO_MODEL"))
    if key not in _model_cache:
        _model_cache[key] = load_model(current_app.config)
    return _model_cache[key]


# -- session ------------------------------------------------------------------
@api.post("/login")
def login():
    body = request.get_json(silent=True) or {}
    user = User.query.filter_by(username=str(body.get("username", "")).strip().lower()).first()
    if user is None or not user.check_password(str(body.get("password", ""))):
        return err("Wrong username or password.", 401)
    return jsonify(token=_serializer().dumps({"uid": user.id}), user=user.public())


@api.get("/config")
def config():
    """Tells the sign-in page whether this is the open public demo."""
    if not current_app.config.get("PUBLIC_DEMO"):
        return jsonify(public_demo=False)
    from . import PUBLIC_DEMO_USERS
    return jsonify(public_demo=True, accounts=[{"username": u, "password": p, "role": r} for u, _, r, p in PUBLIC_DEMO_USERS])


def _sample_volume():
    """A synthetic stand-in for a scan (noise-textured ellipsoid with a dark centre). Not anatomy."""
    rng = np.random.default_rng()
    g = np.meshgrid(*(np.arange(s, dtype=np.float32) for s in (176, 208, 176)), indexing="ij")
    brain = ((g[0] - 88) / 66) ** 2 + ((g[1] - 104) / 84) ** 2 + ((g[2] - 88) / 62) ** 2 <= 1
    r = float(rng.uniform(26, 36))
    hole = ((g[0] - 88) / r) ** 2 + ((g[1] - 104) / (r * 1.5)) ** 2 + ((g[2] - 88) / r) ** 2 <= 1
    vol = np.where(brain, 900 + rng.normal(0, 60, brain.shape), 0)
    return np.where(hole, vol * 0.2, vol).astype(np.float32)


@api.get("/bundle")
def bundle():
    """Instruments and referral rules. Public: it holds no personal data and the
    field app needs it before first sign-in to work offline."""
    return jsonify(load_bundle())


# -- field sync -----------------------------------------------------------------
def _clean_text(value, limit: int) -> str:
    return str(value or "").strip()[:limit]


def _int_or_none(value, lo: int, hi: int):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value) if lo <= value <= hi else None


def _store_screening(rec: dict, user: User) -> dict:
    sid = rec.get("id")
    if not isinstance(sid, str) or not ID_RE.match(sid):
        return {"id": sid, "status": "rejected", "error": "missing or malformed screening id"}
    existing = db.session.get(Screening, sid)
    if existing is not None:                        # safe to send the same record twice
        return {"id": sid, "status": "duplicate", "outcome": existing.outcome}

    consent = rec.get("consent") or {}
    if consent.get("given") is not True:
        return {"id": sid, "status": "rejected", "error": "no recorded consent; nothing was stored"}
    p = rec.get("participant") or {}
    pid, name = p.get("id"), _clean_text(p.get("name"), 120)
    if not isinstance(pid, str) or not ID_RE.match(pid) or not name:
        return {"id": sid, "status": "rejected", "error": "participant id and name are required"}

    participant = db.session.get(Participant, pid)
    if participant is None:
        participant = Participant(id=pid)
        db.session.add(participant)
    participant.code = _clean_text(p.get("code"), 24)
    participant.name = name
    participant.age = _int_or_none(p.get("age"), 18, 120)
    participant.sex = _clean_text(p.get("sex"), 8)
    participant.education_years = _int_or_none(p.get("education_years"), 0, 30)
    participant.village = _clean_text(p.get("village"), 120)
    participant.phone = _clean_text(p.get("phone"), 32)

    # The server never trusts the outcome computed on the device: it scores again.
    result = evaluate(rec)
    record = {k: rec.get(k) for k in ("answers", "red_flags", "conditions", "consent", "informant", "app_version")}
    screening = Screening(id=sid, participant=participant, worker_id=user.id, outcome=result["outcome"],
                          result=result, record=record,
                          client_outcome=_clean_text((rec.get("client_result") or {}).get("outcome"), 16) or None,
                          started_at=_clean_text(rec.get("started_at"), 40),
                          finished_at=_clean_text(rec.get("finished_at"), 40))
    db.session.add(screening)
    if result["outcome"] in ("URGENT", "REFER"):
        db.session.add(Referral(screening=screening, participant=participant, priority=result["outcome"]))
    return {"id": sid, "status": "stored", "outcome": result["outcome"]}


@api.post("/sync")
@require("field", "clinician")
def sync():
    body = request.get_json(silent=True) or {}
    records = body.get("screenings")
    if not isinstance(records, list) or len(records) > 200:
        return err("Send a list of up to 200 screenings.")
    results = []
    for rec in records:
        res = _store_screening(rec, g.user) if isinstance(rec, dict) else {"status": "rejected", "error": "not an object"}
        if res["status"] == "stored":
            db.session.commit()
        else:
            db.session.rollback()
        results.append(res)
    return jsonify(results=results)


@api.get("/stats")
@require("field", "clinician")
def stats():
    by_outcome = dict(db.session.query(Screening.outcome, func.count()).group_by(Screening.outcome).all())
    pending = Referral.query.filter(Referral.status == "pending").count()
    return jsonify(screenings=sum(by_outcome.values()), by_outcome=by_outcome, referrals_pending=pending)


# -- clinic ---------------------------------------------------------------------
@api.get("/referrals")
@require("clinician")
def referrals():
    q = Referral.query
    status = request.args.get("status")
    if status in Referral.STATUSES:
        q = q.filter(Referral.status == status)
    rows = q.order_by(Referral.priority.desc(), Referral.created_at.desc()).limit(500).all()
    return jsonify(referrals=[r.public() for r in rows])


@api.patch("/referrals/<int:rid>")
@require("clinician")
def update_referral(rid: int):
    ref = db.session.get(Referral, rid)
    if ref is None:
        return err("Referral not found.", 404)
    body = request.get_json(silent=True) or {}
    if "status" in body:
        if body["status"] not in Referral.STATUSES:
            return err(f"Status must be one of: {', '.join(Referral.STATUSES)}.")
        ref.status = body["status"]
    if "notes" in body:
        ref.notes = _clean_text(body["notes"], 4000)
    db.session.commit()
    return jsonify(referral=ref.public())


@api.get("/participants/<pid>")
@require("clinician")
def participant(pid: str):
    p = db.session.get(Participant, pid)
    if p is None:
        return err("Person not found.", 404)
    screenings = sorted(p.screenings, key=lambda s: s.received_at, reverse=True)
    studies = MriStudy.query.filter_by(participant_id=pid).order_by(MriStudy.created_at.desc()).all()
    refs = Referral.query.filter_by(participant_id=pid).order_by(Referral.created_at.desc()).all()
    return jsonify(participant=p.public(), screenings=[s.public() for s in screenings],
                   referrals=[{"id": r.id, "status": r.status, "priority": r.priority, "notes": r.notes} for r in refs],
                   mri=[m.public() for m in studies])


@api.get("/model")
@require("field", "clinician")
def model_status():
    model, message = get_model()
    if model is None:
        return jsonify(loaded=False, message=message)
    return jsonify(message=message, **model.status())


def _suffix(filename: str) -> str | None:
    lower = filename.lower()
    return next((s for s in MRI_SUFFIXES if lower.endswith(s)), None)


@api.post("/participants/<pid>/mri")
@require("clinician")
def upload_mri(pid: str):
    p = db.session.get(Participant, pid)
    if p is None:
        return err("Person not found.", 404)
    model, message = get_model()
    if model is None:
        return err(message, 503)
    files = [f for f in request.files.getlist("files") if f and f.filename]
    kinds = {_suffix(f.filename) for f in files}
    if None in kinds or not files:
        return err("Upload a .nii or .nii.gz file, or an Analyze pair (.hdr and .img together).")
    if kinds not in ({".nii"}, {".nii.gz"}, {".img", ".hdr"}) or len(files) != len(kinds):
        return err("Upload one NIfTI file, or exactly one .hdr and one .img that belong together.")

    folder = Path(current_app.config["UPLOAD_DIR"]) / uuid.uuid4().hex
    folder.mkdir(parents=True)
    main = None
    for f in files:                                  # never reuse the client's file name on disk
        suffix = _suffix(f.filename)
        dest = folder / f"scan{suffix}"
        f.save(dest)
        if suffix != ".hdr":
            main = dest

    study = MriStudy(participant_id=pid, uploaded_by=g.user.id, filename=_clean_text(files[0].filename, 255),
                     stored_path=str(main), status="scored")
    try:
        x, info = model.prepare(main)
        study.result = {**model.predict(x), "input": info}
        np.save(folder / "input.npy", x.astype(np.float16))
    except InputRejected as e:
        study.status, study.result = "rejected", {"reason": str(e)}
    except Exception as e:  # unreadable or corrupt file
        study.status, study.result = "rejected", {"reason": f"The file could not be read as an MRI volume ({type(e).__name__})."}
    db.session.add(study)
    db.session.commit()
    return jsonify(study=study.public()), (201 if study.status == "scored" else 422)


@api.post("/scan")
@require("clinician")
def scan_check():
    """Stand-alone scan check: nothing is stored. Returns measurements, not a diagnosis."""
    import base64
    import shutil
    import tempfile

    try:
        age = float(request.form.get("age", ""))
    except ValueError:
        return err("Enter the person's age in years.")
    if not 18 <= age <= 110:
        return err("Enter an age between 18 and 110.")
    sample = request.form.get("sample") == "1"
    files = [f for f in request.files.getlist("files") if f and f.filename]
    kinds = {_suffix(f.filename) for f in files}
    if not sample and (not files or None in kinds or kinds not in ({".nii"}, {".nii.gz"}, {".img", ".hdr"}) or len(files) != len(kinds)):
        return err("Upload one NIfTI file, or exactly one .hdr and one .img that belong together.")

    folder = Path(tempfile.mkdtemp(dir=current_app.config["UPLOAD_DIR"]))
    try:
        main = None
        for f in ([] if sample else files):
            suffix = _suffix(f.filename)
            f.save(folder / f"scan{suffix}")
            if suffix != ".hdr":
                main = folder / f"scan{suffix}"
        cfg = PreprocessConfig()
        try:
            vol, zooms = (_sample_volume(), (1.0, 1.0, 1.0)) if sample else load_volume(main)
            checks = check_input(vol, zooms, cfg)
        except InputRejected as e:
            return jsonify(accepted=False, reason=str(e)), 422
        except Exception as e:
            return jsonify(accepted=False, reason=f"The file could not be read as an MRI volume ({type(e).__name__})."), 422
        x = preprocess(vol, cfg)
        out = {"accepted": True, "sample": sample, "checks": checks, "measure": assess(vol, age),
               "reference": {k: load_reference()[k] for k in ("name", "validation", "limits", "reference_group")},
               "image": "data:image/png;base64," + base64.b64encode(render_slices(x)).decode()}
        model, message = get_model()
        if model is not None:
            out["model"] = {**model.predict(x), "evaluation": model.meta.get("evaluation")}
        else:
            out["model_message"] = message
        return jsonify(out)
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def _study_array(sid: int):
    study = db.session.get(MriStudy, sid)
    if study is None or study.status != "scored":
        return None, None
    folder = Path(study.stored_path).parent
    return study, folder


@api.get("/mri/<int:sid>/image")
@require("clinician")
def mri_image(sid: int):
    study, folder = _study_array(sid)
    if study is None:
        return err("No image for this study.", 404)
    x = np.load(folder / "input.npy").astype(np.float32)
    heat = np.load(folder / "heat.npy") if (folder / "heat.npy").exists() and request.args.get("overlay") == "1" else None
    return Response(render_slices(x, heat), mimetype="image/png", headers={"Cache-Control": "no-store"})


@api.post("/mri/<int:sid>/explain")
@require("clinician")
def mri_explain(sid: int):
    study, folder = _study_array(sid)
    model, message = get_model()
    if study is None:
        return err("No image for this study.", 404)
    if model is None:
        return err(message, 503)
    if not (folder / "heat.npy").exists():
        x = np.load(folder / "input.npy").astype(np.float32)
        np.save(folder / "heat.npy", model.occlusion_map(x))
    return jsonify(ready=True)
