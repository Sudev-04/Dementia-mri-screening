"""Flask application: sync API for the field app, referral queue and MRI analysis for the clinic."""
from __future__ import annotations

import os
import secrets
from pathlib import Path

from flask import Flask, jsonify, send_from_directory

from .models import db

ROOT = Path(__file__).resolve().parent.parent
PUBLIC_DEMO_USERS = [("demo-field", "Demo field worker", "field", "demo-field-2026"),
                     ("demo-clinic", "Demo clinician", "clinician", "demo-clinic-2026")]


def _secret_key(instance: Path) -> str:
    env = os.environ.get("ALZ_SECRET_KEY")
    if env:
        return env
    f = instance / "secret_key"
    if not f.exists():
        f.write_text(secrets.token_hex(32))
    return f.read_text().strip()


def create_app(config: dict | None = None) -> Flask:
    app = Flask(__name__, static_folder="static", static_url_path="/static",
                instance_path=str((config or {}).get("INSTANCE_PATH") or os.environ.get("ALZ_INSTANCE") or ROOT / "instance"))
    instance = Path(app.instance_path)
    instance.mkdir(parents=True, exist_ok=True)

    app.config.update(
        SQLALCHEMY_DATABASE_URI=os.environ.get("ALZ_DATABASE_URL") or f"sqlite:///{instance / 'alzscreen.db'}",
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
        MAX_CONTENT_LENGTH=200 * 1024 * 1024,
        UPLOAD_DIR=str(instance / "uploads"),
        MODEL_DIR=os.environ.get("ALZ_MODEL_DIR") or str(ROOT / "models" / "current"),
        DEMO_MODEL_DIR=str(ROOT / "models" / "demo"),
        ALLOW_DEMO_MODEL=os.environ.get("ALZ_ALLOW_DEMO_MODEL") == "1",
        TOKEN_MAX_AGE=30 * 24 * 3600,
        PUBLIC_DEMO=os.environ.get("ALZ_PUBLIC_DEMO") == "1",
    )
    app.config.update(config or {})
    if not app.config.get("SECRET_KEY"):
        app.config["SECRET_KEY"] = _secret_key(instance)
    Path(app.config["UPLOAD_DIR"]).mkdir(parents=True, exist_ok=True)

    db.init_app(app)
    if app.config["PUBLIC_DEMO"]:
        app.config["ALLOW_DEMO_MODEL"] = True
    with app.app_context():
        db.create_all()
        if app.config["PUBLIC_DEMO"]:
            # Open demo: the passwords are shown on the sign-in page on purpose.
            from .models import User
            for username, name, role, password in PUBLIC_DEMO_USERS:
                if User.query.filter_by(username=username).first() is None:
                    u = User(username=username, name=name, role=role)
                    u.set_password(password)
                    db.session.add(u)
            db.session.commit()

    from .api import api
    app.register_blueprint(api, url_prefix="/api")

    @app.get("/")
    def index():
        return send_from_directory(app.static_folder, "index.html")

    @app.get("/sw.js")
    def service_worker():
        # Served from the root so its scope covers the whole app.
        resp = send_from_directory(app.static_folder, "sw.js", mimetype="text/javascript")
        resp.headers["Cache-Control"] = "no-cache"
        resp.headers["Service-Worker-Allowed"] = "/"
        return resp

    @app.after_request
    def security_headers(resp):
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        resp.headers.setdefault("Referrer-Policy", "no-referrer")
        resp.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' blob: data:; style-src 'self'; script-src 'self'; "
            "connect-src 'self'; frame-ancestors 'none'")
        return resp

    @app.errorhandler(413)
    def too_large(_):
        return jsonify(error="That file is larger than the 200 MB upload limit."), 413

    return app
