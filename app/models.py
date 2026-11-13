"""Database tables. SQLite by default; set ALZ_DATABASE_URL for PostgreSQL."""
from __future__ import annotations

from datetime import datetime, timezone

from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import check_password_hash, generate_password_hash

db = SQLAlchemy()


def now() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:          # SQLite returns naive datetimes; they were stored as UTC
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    name = db.Column(db.String(120), nullable=False)
    role = db.Column(db.String(16), nullable=False)          # field | clinician
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime(timezone=True), default=now)

    def set_password(self, password: str) -> None:
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)

    def public(self) -> dict:
        return {"username": self.username, "name": self.name, "role": self.role}


class Participant(db.Model):
    id = db.Column(db.String(64), primary_key=True)           # UUID made on the field device
    code = db.Column(db.String(24), index=True)
    name = db.Column(db.String(120), nullable=False)
    age = db.Column(db.Integer)
    sex = db.Column(db.String(8))
    education_years = db.Column(db.Integer)
    village = db.Column(db.String(120))
    phone = db.Column(db.String(32))
    created_at = db.Column(db.DateTime(timezone=True), default=now)

    def public(self) -> dict:
        return {"id": self.id, "code": self.code, "name": self.name, "age": self.age, "sex": self.sex,
                "education_years": self.education_years, "village": self.village, "phone": self.phone}


class Screening(db.Model):
    id = db.Column(db.String(64), primary_key=True)           # UUID made on the field device
    participant_id = db.Column(db.String(64), db.ForeignKey("participant.id"), nullable=False, index=True)
    worker_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    outcome = db.Column(db.String(16), nullable=False, index=True)
    result = db.Column(db.JSON, nullable=False)                # server-computed scores and reasons
    record = db.Column(db.JSON, nullable=False)                # raw answers, consent, conditions
    client_outcome = db.Column(db.String(16))                  # what the device showed, for auditing
    started_at = db.Column(db.String(40))
    finished_at = db.Column(db.String(40))
    received_at = db.Column(db.DateTime(timezone=True), default=now)

    participant = db.relationship(Participant, backref="screenings")
    worker = db.relationship(User)

    def public(self) -> dict:
        return {"id": self.id, "outcome": self.outcome, "result": self.result,
                "answers": (self.record or {}).get("answers"), "informant": (self.record or {}).get("informant"),
                "finished_at": self.finished_at, "received_at": iso(self.received_at),
                "worker": self.worker.name if self.worker else None,
                "outcome_mismatch": bool(self.client_outcome and self.client_outcome != self.outcome)}


class Referral(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    screening_id = db.Column(db.String(64), db.ForeignKey("screening.id"), unique=True, nullable=False)
    participant_id = db.Column(db.String(64), db.ForeignKey("participant.id"), nullable=False, index=True)
    priority = db.Column(db.String(16), nullable=False)        # URGENT | REFER
    status = db.Column(db.String(16), nullable=False, default="pending")
    notes = db.Column(db.Text, default="")
    created_at = db.Column(db.DateTime(timezone=True), default=now)
    updated_at = db.Column(db.DateTime(timezone=True), default=now, onupdate=now)

    screening = db.relationship(Screening, backref=db.backref("referral", uselist=False))
    participant = db.relationship(Participant)

    STATUSES = ("pending", "seen", "scan_ordered", "closed")

    def public(self) -> dict:
        return {"id": self.id, "priority": self.priority, "status": self.status, "notes": self.notes,
                "created_at": iso(self.created_at), "updated_at": iso(self.updated_at),
                "participant": self.participant.public(),
                "screening": {"id": self.screening.id, "outcome": self.screening.outcome,
                              "reasons": self.screening.result.get("reasons", []),
                              "notes": self.screening.result.get("notes", []),
                              "scores": self.screening.result.get("scores", {}),
                              "finished_at": self.screening.finished_at}}


class MriStudy(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    participant_id = db.Column(db.String(64), db.ForeignKey("participant.id"), nullable=False, index=True)
    uploaded_by = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    filename = db.Column(db.String(255))
    stored_path = db.Column(db.String(500))
    status = db.Column(db.String(16), nullable=False)          # scored | rejected
    result = db.Column(db.JSON)
    created_at = db.Column(db.DateTime(timezone=True), default=now)

    def public(self) -> dict:
        return {"id": self.id, "filename": self.filename, "status": self.status,
                "result": self.result, "created_at": iso(self.created_at)}
