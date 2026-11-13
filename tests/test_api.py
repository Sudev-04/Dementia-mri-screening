import io
import uuid
from pathlib import Path

import pytest

from app import create_app
from app.models import Referral, Screening, User, db

ROOT = Path(__file__).resolve().parent.parent
POSITIVE = {"csid_cog": {"elbow": 1, "hammer": 1, "store": 0, "day": 0, "season": 0, "point": 1, "recall": 0}}
NEGATIVE = {"csid_cog": {"elbow": 1, "hammer": 1, "store": 1, "day": 1, "season": 1, "point": 1, "recall": 3}}


@pytest.fixture()
def app(tmp_path):
    app = create_app({"INSTANCE_PATH": str(tmp_path), "TESTING": True, "SECRET_KEY": "test",
                      "MODEL_DIR": str(tmp_path / "no-model"), "ALLOW_DEMO_MODEL": False})
    with app.app_context():
        for username, role in (("field1", "field"), ("clinic1", "clinician")):
            u = User(username=username, name=username, role=role)
            u.set_password("correct-horse")
            db.session.add(u)
        db.session.commit()
    return app


@pytest.fixture()
def client(app):
    return app.test_client()


def token(client, username):
    r = client.post("/api/login", json={"username": username, "password": "correct-horse"})
    assert r.status_code == 200
    return {"Authorization": f"Bearer {r.get_json()['token']}"}


def record(answers, **extra):
    return {"id": str(uuid.uuid4()), "consent": {"given": True, "by": "participant", "at": "2026-11-11T10:00:00Z"},
            "participant": {"id": str(uuid.uuid4()), "code": "P-TEST01", "name": "Test Person", "age": 72,
                            "sex": "F", "education_years": 4, "village": "Ward 3"},
            "answers": answers, "red_flags": [], "conditions": [], **extra}


def test_login_rejects_bad_password(client):
    assert client.post("/api/login", json={"username": "field1", "password": "nope"}).status_code == 401


def test_api_needs_a_token(client):
    assert client.post("/api/sync", json={"screenings": []}).status_code == 401
    assert client.get("/api/referrals").status_code == 401
    assert client.get("/api/referrals", headers={"Authorization": "Bearer forged"}).status_code == 401


def test_bundle_is_public_and_complete(client):
    data = client.get("/api/bundle").get_json()
    assert set(data["instruments"]) == {"csid_cog", "csid_inf", "iqcode16", "phq2"}
    assert data["rules"]["thresholds"]["csid_combined_max"]["value"] == 4


def test_sync_stores_scores_and_creates_referral(app, client):
    h = token(client, "field1")
    pos, neg = record(POSITIVE), record(NEGATIVE)
    res = client.post("/api/sync", json={"screenings": [pos, neg]}, headers=h).get_json()["results"]
    assert [r["status"] for r in res] == ["stored", "stored"]
    assert [r["outcome"] for r in res] == ["REFER", "NO_REFERRAL"]
    with app.app_context():
        assert Referral.query.count() == 1
        assert db.session.get(Screening, pos["id"]).result["scores"]["csid_cog"] == 3


def test_sync_is_idempotent(app, client):
    h = token(client, "field1")
    rec = record(POSITIVE)
    client.post("/api/sync", json={"screenings": [rec]}, headers=h)
    again = client.post("/api/sync", json={"screenings": [rec]}, headers=h).get_json()["results"][0]
    assert again["status"] == "duplicate"
    with app.app_context():
        assert Screening.query.count() == 1 and Referral.query.count() == 1


def test_server_rescoring_overrides_the_device(app, client):
    h = token(client, "field1")
    rec = record(POSITIVE, client_result={"outcome": "NO_REFERRAL"})     # tampered or stale device
    out = client.post("/api/sync", json={"screenings": [rec]}, headers=h).get_json()["results"][0]
    assert out["outcome"] == "REFER"
    with app.app_context():
        assert db.session.get(Screening, rec["id"]).public()["outcome_mismatch"] is True


def test_no_consent_means_nothing_is_stored(app, client):
    h = token(client, "field1")
    rec = record(POSITIVE); rec["consent"]["given"] = False
    out = client.post("/api/sync", json={"screenings": [rec]}, headers=h).get_json()["results"][0]
    assert out["status"] == "rejected"
    with app.app_context():
        assert Screening.query.count() == 0


def test_one_bad_record_does_not_block_the_rest(client):
    h = token(client, "field1")
    bad = record(POSITIVE); bad["participant"]["name"] = ""
    res = client.post("/api/sync", json={"screenings": [bad, record(NEGATIVE)]}, headers=h).get_json()["results"]
    assert [r["status"] for r in res] == ["rejected", "stored"]


def test_field_worker_cannot_open_clinic_pages(client):
    h = token(client, "field1")
    assert client.get("/api/referrals", headers=h).status_code == 403
    assert client.get("/api/participants/whatever", headers=h).status_code == 403


def test_referral_workflow(client):
    rec = record(POSITIVE)
    client.post("/api/sync", json={"screenings": [rec]}, headers=token(client, "field1"))
    h = token(client, "clinic1")
    refs = client.get("/api/referrals", headers=h).get_json()["referrals"]
    assert len(refs) == 1 and refs[0]["participant"]["name"] == "Test Person"
    rid = refs[0]["id"]
    assert client.patch(f"/api/referrals/{rid}", json={"status": "bogus"}, headers=h).status_code == 400
    upd = client.patch(f"/api/referrals/{rid}", json={"status": "scan_ordered", "notes": "MRI booked"}, headers=h).get_json()
    assert upd["referral"]["status"] == "scan_ordered"
    person = client.get(f"/api/participants/{rec['participant']['id']}", headers=h).get_json()
    assert person["screenings"][0]["outcome"] == "REFER" and person["referrals"][0]["notes"] == "MRI booked"


def test_mri_upload_without_model_says_so(client):
    rec = record(POSITIVE)
    client.post("/api/sync", json={"screenings": [rec]}, headers=token(client, "field1"))
    h = token(client, "clinic1")
    assert client.get("/api/model", headers=h).get_json()["loaded"] is False
    r = client.post(f"/api/participants/{rec['participant']['id']}/mri", headers=h,
                    data={"files": (io.BytesIO(b"x"), "scan.nii")}, content_type="multipart/form-data")
    assert r.status_code == 503 and "No trained model" in r.get_json()["error"]


@pytest.mark.skipif(not (ROOT / "models" / "demo" / "model_meta.json").exists(), reason="demo model not present")
def test_mri_flow_with_demo_model(tmp_path, t88_file):
    app = create_app({"INSTANCE_PATH": str(tmp_path / "inst"), "TESTING": True, "SECRET_KEY": "test",
                      "MODEL_DIR": str(tmp_path / "no-model"), "ALLOW_DEMO_MODEL": True})
    with app.app_context():
        for username, role in (("field1", "field"), ("clinic1", "clinician")):
            u = User(username=username, name=username, role=role); u.set_password("correct-horse"); db.session.add(u)
        db.session.commit()
    c = app.test_client()
    rec = record(POSITIVE)
    c.post("/api/sync", json={"screenings": [rec]}, headers=token(c, "field1"))
    h = token(c, "clinic1")
    pid = rec["participant"]["id"]
    assert c.get("/api/model", headers=h).get_json()["synthetic"] is True

    files = [(open(t88_file.with_suffix(".hdr"), "rb"), "patient.hdr"), (open(t88_file, "rb"), "patient.img")]
    r = c.post(f"/api/participants/{pid}/mri", headers=h, data={"files": files}, content_type="multipart/form-data")
    assert r.status_code == 201, r.get_json()
    result = r.get_json()["study"]["result"]
    assert 0 <= result["probability"] <= 1 and result["synthetic"] is True and len(result["member_probabilities"]) >= 2
    sid = r.get_json()["study"]["id"]
    img = c.get(f"/api/mri/{sid}/image", headers=h)
    assert img.status_code == 200 and img.data[:8] == b"\x89PNG\r\n\x1a\n"

    # a file that is not a brain volume is refused with a reason, not scored
    bad = c.post(f"/api/participants/{pid}/mri", headers=h, data={"files": (io.BytesIO(b"not an image"), "x.nii")},
                 content_type="multipart/form-data")
    assert bad.status_code == 422 and bad.get_json()["study"]["status"] == "rejected"
    # mismatched upload
    mixed = c.post(f"/api/participants/{pid}/mri", headers=h, data={"files": (io.BytesIO(b"x"), "only.hdr")},
                   content_type="multipart/form-data")
    assert mixed.status_code == 400


def test_scan_check_measures_without_storing(app, client, t88_file):
    from app.models import MriStudy
    h = token(client, "clinic1")
    files = lambda: [(open(t88_file.with_suffix(".hdr"), "rb"), "p.hdr"), (open(t88_file, "rb"), "p.img")]
    r = client.post("/api/scan", headers=h, data={"files": files(), "age": "74"}, content_type="multipart/form-data")
    body = r.get_json()
    assert r.status_code == 200 and body["accepted"] is True
    assert 0 < body["measure"]["brain_fraction"] <= 1 and body["measure"]["band"] in ("usual", "lower", "well_below")
    assert body["image"].startswith("data:image/png;base64,") and "model_message" in body   # no model installed here
    assert client.post("/api/scan", headers=h, data={"files": files()}, content_type="multipart/form-data").status_code == 400
    bad = client.post("/api/scan", headers=h, data={"files": (io.BytesIO(b"junk"), "x.nii"), "age": "70"}, content_type="multipart/form-data")
    assert bad.status_code == 422 and bad.get_json()["accepted"] is False
    assert client.post("/api/scan", headers=token(client, "field1"), data={"age": "70"}).status_code == 403
    with app.app_context():
        assert MriStudy.query.count() == 0
    assert not any(p.is_dir() for p in Path(app.config["UPLOAD_DIR"]).iterdir())            # temporary upload removed


def test_public_demo_mode(tmp_path):
    app = create_app({"INSTANCE_PATH": str(tmp_path), "TESTING": True, "SECRET_KEY": "t", "PUBLIC_DEMO": False})
    assert app.test_client().get("/api/config").get_json() == {"public_demo": False}
    import os
    os.environ["ALZ_PUBLIC_DEMO"] = "1"
    try:
        demo = create_app({"INSTANCE_PATH": str(tmp_path / "demo"), "TESTING": True, "SECRET_KEY": "t"})
    finally:
        del os.environ["ALZ_PUBLIC_DEMO"]
    c = demo.test_client()
    cfg = c.get("/api/config").get_json()
    assert cfg["public_demo"] is True and len(cfg["accounts"]) == 2
    clinic = next(a for a in cfg["accounts"] if a["role"] == "clinician")
    tok = c.post("/api/login", json={"username": clinic["username"], "password": clinic["password"]}).get_json()["token"]
    r = c.post("/api/scan", headers={"Authorization": f"Bearer {tok}"}, data={"age": "72", "sample": "1"})
    body = r.get_json()
    assert r.status_code == 200 and body["sample"] is True and body["measure"]["brain_fraction"] < 1
    assert body["model"]["synthetic"] is True
