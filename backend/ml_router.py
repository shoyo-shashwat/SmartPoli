"""
SmartPoli: Digital Twin endpoints (demo copy; change plan, Change 1).

Switched OFF unless SMARTPOLI_TWIN_ENABLED=1, so the live app never shows twin output. When off, every endpoint
here answers 404 as if it did not exist. Access reuses the app's existing checks: a doctor only reaches patients
with an active link. The twin is a risk flag for the doctor, never a diagnosis or a treatment change; every
doctor action is written to AuditLog so it appears on the patient timeline.
"""
import json
import os
from datetime import datetime, timedelta
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from auth import has_read_access, require_doctor_role, require_patient_read_access
from db import get_db_session, log_audit, AuditLog, DoctorLink, GlucoseReading, HeartRateReading, Patient, TwinEvent, TwinProfile, User

router = APIRouter(tags=["twin"])

REVIEW_WINDOW = timedelta(minutes=60)


def enabled() -> bool:
    return os.getenv("SMARTPOLI_TWIN_ENABLED") == "1"


def _require_enabled():
    if not enabled():
        raise HTTPException(404, "Not found")


def _profile_dict(p: TwinProfile) -> dict:
    d = {c: getattr(p, c) for c in ("sex", "age", "bmi", "diabetes_years", "hba1c", "egfr", "sensor_note", "confidence", "confidence_reason")}
    for c in ("hypertension", "on_insulin", "on_sulfonylurea", "on_metformin", "heart_disease"):
        d[c] = bool(getattr(p, c))
    return d


def _load_patient_series(db: Session, patient_id: int):
    grid = {r.timestamp: r.value for r in db.query(GlucoseReading).filter(GlucoseReading.patient_id == patient_id).all()}
    events = [(e.timestamp, e.kind, e.iu, e.text) for e in db.query(TwinEvent).filter(TwinEvent.patient_id == patient_id).all()]
    return grid, events


def _parse_at(at: Optional[str], profile: TwinProfile, grid) -> datetime:
    if at:
        try:
            return datetime.fromisoformat(at.replace("Z", "")).replace(tzinfo=None)
        except ValueError:
            raise HTTPException(422, "at must be an ISO date and time, e.g. 2020-11-10T09:52:00")
    return profile.replay_start or max(grid)


def _review_state(db: Session, patient_id: int, now: datetime):
    """'reviewed' / 'snoozed' for 60 minutes of replay time after the doctor's action, unless reopened since."""
    rows = (db.query(AuditLog).filter(AuditLog.patient_id == patient_id, AuditLog.action.in_(["twin_reviewed", "twin_snoozed", "twin_reopened"]))
            .order_by(AuditLog.id.desc()).all())
    for r in rows:
        try:
            t = datetime.fromisoformat(json.loads(r.detail or "{}").get("data_time", ""))
        except (ValueError, TypeError):
            continue
        if t <= now:
            if r.action == "twin_reopened" or now - t > REVIEW_WINDOW:
                return "open", None
            return r.action.replace("twin_", ""), {"by": r.actor, "data_time": t.isoformat()}
    return "open", None


def _activity(db: Session, patient_id: int, limit: int = 12):
    rows = (db.query(AuditLog).filter(AuditLog.patient_id == patient_id, AuditLog.action.in_(["twin_reviewed", "twin_snoozed", "twin_reopened", "clinical_note"]))
            .order_by(AuditLog.id.desc()).limit(limit).all())
    label = {"twin_reviewed": "Marked as reviewed", "twin_snoozed": "Alert snoozed for 1 hour", "twin_reopened": "Reopened", "clinical_note": "Note added"}
    out = []
    for r in rows:
        extra = ""
        if r.action == "clinical_note" and r.detail:
            extra = ": " + (r.detail[:80] + ("..." if len(r.detail) > 80 else ""))
        out.append({"at": r.at.isoformat() if r.at else None, "by": (r.actor or "").split(":")[-1], "text": label[r.action] + extra})
    return out


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if hasattr(x, "item"):          # numpy scalar
        return x.item()
    return x


@router.get("/twin/info")
def twin_info():
    """Public and tiny: tells the pages whether the twin exists on this copy of the app (404 on the live app)."""
    _require_enabled()
    out = {"enabled": True}
    if os.getenv("SMARTPOLI_DEMO_LOGIN") == "1":
        out["demo_login"] = {"email": os.getenv("SMARTPOLI_DEMO_EMAIL", "doctor@smartpoli.demo"), "password": os.getenv("SMARTPOLI_DEMO_PASSWORD", "demo1234")}
    return out


@router.get("/patients/{patient_id}/ml")
def get_twin(patient_id: int, at: Optional[str] = None, user: User = Depends(require_patient_read_access), db: Session = Depends(get_db_session)):
    _require_enabled()
    import twin_service
    profile = db.query(TwinProfile).filter(TwinProfile.patient_id == patient_id).first()
    if not profile:
        raise HTTPException(404, "No sugar sensor data for this patient, so there is no twin.")
    grid, events = _load_patient_series(db, patient_id)
    if not grid:
        raise HTTPException(404, "No sugar readings for this patient.")
    when = _parse_at(at, profile, grid)
    try:
        out = twin_service.predict(_profile_dict(profile), grid, events, when)
    except ValueError as e:
        raise HTTPException(422, str(e))
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    now = datetime.fromisoformat(out["as_of"])
    hr = (db.query(HeartRateReading).filter(HeartRateReading.patient_id == patient_id, HeartRateReading.timestamp > now - timedelta(hours=6), HeartRateReading.timestamp <= now)
          .order_by(HeartRateReading.timestamp).all())
    out["heart_rate"] = [[h.timestamp.isoformat(), round(h.bpm)] for h in hr]
    state, info = _review_state(db, patient_id, now)
    out.update({"patient_id": patient_id, "label": profile.label or patient.name, "summary": profile.summary, "review": {"state": state, "info": info},
                "activity": _activity(db, patient_id), "data_source": profile.data_source})
    return _jsonable(out)


@router.get("/twin/worklist")
def twin_worklist(at: Optional[str] = None, user: User = Depends(require_doctor_role), db: Session = Depends(get_db_session)):
    """Patients with twin data linked to this doctor, each at its default replay moment, most urgent first."""
    _require_enabled()
    import twin_service
    ids = [r.patient_id for r in db.query(DoctorLink).filter(DoctorLink.doctor_user_id == user.id, DoctorLink.status == "active").all()]
    rows = []
    for pid in ids:
        profile = db.query(TwinProfile).filter(TwinProfile.patient_id == pid).first()
        if not profile:
            continue
        grid, events = _load_patient_series(db, pid)
        if not grid:
            continue
        r = twin_service.predict(_profile_dict(profile), grid, events, _parse_at(at, profile, grid), history_minutes=0)
        state, _ = _review_state(db, pid, datetime.fromisoformat(r["as_of"]))
        rows.append({"patient_id": pid, "label": profile.label, "summary": profile.summary, "as_of": r["as_of"], "headline": r["headline"],
                     "tier": r["low"]["tier"], "review": state, "high_tier": r["high"]["tier"], "confidence": r["confidence"], "data_source": profile.data_source})
    order = {"alert": 0, "watch": 1, "calm": 2}
    rows.sort(key=lambda x: (order[x["tier"]] + (3 if x["review"] != "open" else 0), x["label"] or ""))
    return {"patients": rows}


class TwinAction(BaseModel):
    action: Literal["reviewed", "snoozed", "reopened"]
    at: str


@router.post("/patients/{patient_id}/ml/actions")
def twin_action(patient_id: int, body: TwinAction, user: User = Depends(require_doctor_role), db: Session = Depends(get_db_session)):
    _require_enabled()
    if not has_read_access(db, user, patient_id):
        raise HTTPException(403, "You do not have access to this patient's records.")
    try:
        when = datetime.fromisoformat(body.at.replace("Z", "")).replace(tzinfo=None)
    except ValueError:
        raise HTTPException(422, "at must be an ISO date and time")
    log_audit(db, patient_id, user.name or "doctor", f"twin_{body.action}", json.dumps({"data_time": when.isoformat()}))
    state, info = _review_state(db, patient_id, when)
    return {"review": {"state": state, "info": info}, "activity": _activity(db, patient_id)}
