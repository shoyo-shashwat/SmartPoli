"""Digital Twin endpoints: off by default, access-controlled, and a working review loop (synthetic readings)."""
import math
import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from main import app  # noqa: E402
from conftest import register_and_login  # noqa: E402
from db import SessionLocal, DoctorLink, GlucoseReading, TwinEvent, TwinProfile  # noqa: E402

START = datetime(2026, 1, 5, 0, 0)


def _linked_patient(client, with_twin=True):
    register_and_login(client, role="patient")
    patient_auth = client.headers["Authorization"]
    pid = client.post("/patients", json={"name": "Twin Test"}).json()["id"]
    code = client.post(f"/patients/{pid}/doctor-links").json()["code"]
    register_and_login(client, role="doctor", name="Dr. Twin")
    assert client.post("/doctor/link/redeem", json={"code": code}).status_code == 200
    if with_twin:
        db = SessionLocal()
        try:
            db.add(TwinProfile(patient_id=pid, label="Demo patient T", summary="Woman, 60. Insulin", age=60, sex=1, bmi=25, diabetes_years=10, hba1c=60, egfr=90,
                               hypertension=False, on_insulin=True, heart_disease=True, replay_start=START + timedelta(days=2, hours=9)))
            for k in range(3 * 96):
                t = START + timedelta(minutes=15 * k)
                db.add(GlucoseReading(patient_id=pid, timestamp=t, value=120 + 50 * math.sin(k / 9.0), source="replay"))
            db.add(TwinEvent(patient_id=pid, timestamp=START + timedelta(days=2, hours=7), kind="meal", text="Breakfast"))
            db.add(TwinEvent(patient_id=pid, timestamp=START + timedelta(days=2, hours=7), kind="insulin", text="Insulin, 8 IU", iu=8.0))
            db.commit()
        finally:
            db.close()
    return pid, patient_auth


def test_everything_is_404_when_the_twin_is_off(monkeypatch):
    monkeypatch.delenv("SMARTPOLI_TWIN_ENABLED", raising=False)
    with TestClient(app) as client:
        pid, _ = _linked_patient(client)
        assert client.get("/twin/info").status_code == 404
        assert client.get(f"/patients/{pid}/ml").status_code == 404
        assert client.get("/twin/worklist").status_code == 404
        assert client.post(f"/patients/{pid}/ml/actions", json={"action": "reviewed", "at": "2026-01-07T09:00:00"}).status_code == 404


def test_twin_response_and_review_loop(monkeypatch):
    monkeypatch.setenv("SMARTPOLI_TWIN_ENABLED", "1")
    with TestClient(app) as client:
        pid, _ = _linked_patient(client)
        assert client.get("/twin/info").json()["enabled"] is True
        r = client.get(f"/patients/{pid}/ml")
        assert r.status_code == 200, r.text
        j = r.json()
        assert j["low"]["tier"] in ("calm", "watch", "alert") and j["high"]["tier"] in ("calm", "watch", "alert")
        assert len(j["forecast"]) == 8 and all(p["low"] <= p["mg_dl"] <= p["high"] for p in j["forecast"])
        assert j["history"] and j["model_version"] and "diagnosis" in j["disclaimer"]
        assert j["low"]["reasons"] and j["review"]["state"] == "open"
        at = j["as_of"]
        assert client.get("/twin/worklist").json()["patients"][0]["patient_id"] == pid
        r = client.post(f"/patients/{pid}/ml/actions", json={"action": "reviewed", "at": at}).json()
        assert r["review"]["state"] == "reviewed" and "Marked as reviewed" in r["activity"][0]["text"]
        assert client.get(f"/patients/{pid}/ml?at={at}").json()["review"]["state"] == "reviewed"
        later = (datetime.fromisoformat(at) + timedelta(minutes=90)).isoformat()
        assert client.get(f"/patients/{pid}/ml?at={later}").json()["review"]["state"] == "open"      # review lapses after an hour
        r = client.post(f"/patients/{pid}/ml/actions", json={"action": "reopened", "at": at}).json()
        assert r["review"]["state"] == "open"


def test_unlinked_doctor_is_refused(monkeypatch):
    monkeypatch.setenv("SMARTPOLI_TWIN_ENABLED", "1")
    with TestClient(app) as client:
        pid, _ = _linked_patient(client)
        register_and_login(client, role="doctor", name="Dr. Stranger")
        assert client.get(f"/patients/{pid}/ml").status_code == 403
        assert client.post(f"/patients/{pid}/ml/actions", json={"action": "reviewed", "at": "2026-01-07T09:00:00"}).status_code == 403
        assert client.get("/twin/worklist").json()["patients"] == []


def test_patient_without_sensor_data_has_no_twin(monkeypatch):
    monkeypatch.setenv("SMARTPOLI_TWIN_ENABLED", "1")
    with TestClient(app) as client:
        pid, _ = _linked_patient(client, with_twin=False)
        assert client.get(f"/patients/{pid}/ml").status_code == 404


def test_demo_login_info_only_when_asked(monkeypatch):
    monkeypatch.setenv("SMARTPOLI_TWIN_ENABLED", "1")
    monkeypatch.delenv("SMARTPOLI_DEMO_LOGIN", raising=False)
    with TestClient(app) as client:
        assert "demo_login" not in client.get("/twin/info").json()
        monkeypatch.setenv("SMARTPOLI_DEMO_LOGIN", "1")
        assert client.get("/twin/info").json()["demo_login"]["email"] == "doctor@smartpoli.demo"
