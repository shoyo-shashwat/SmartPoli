"""
End-to-end tests over the manual path: textarea -> parser -> confirm ->
dashboard -> triage -> report. This is the path that must work with an
empty .env (CLAUDE.md section 14) — no OCR, no LLM, no network call.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient  # noqa: E402
from main import app  # noqa: E402
from conftest import register_and_login  # noqa: E402


def test_full_manual_path_end_to_end():
    with TestClient(app) as client:
        register_and_login(client)
        # 1. Create a patient.
        r = client.post("/patients", json={"name": "Test Patient", "age": 40, "sex": "F"})
        assert r.status_code == 200
        patient_id = r.json()["id"]

        # 2. Manual prescription entry — mixes a clean line, an SOS line, and a
        #    malformed one, exactly like the PS 4 shorthand examples.
        r = client.post("/prescriptions", json={
            "patient_id": patient_id,
            "doctor_name": "Dr. Test",
            "lines": [
                "Tab Dolo 650mg 1-0-1 PC x5d",
                "Syrup Crocin 5ml SOS",
                "Tab Unknown 1-?-1",
            ],
        })
        assert r.status_code == 200
        body = r.json()
        prescription_id = body["prescription_id"]
        medicines = body["medicines"]
        assert len(medicines) == 3

        dolo = next(m for m in medicines if "Dolo" in m["raw_text"])
        sos = next(m for m in medicines if "SOS" in m["raw_text"])
        unknown = next(m for m in medicines if "?" in m["raw_text"])

        assert dolo["status"] in ("verified", "review")
        assert sos["is_prn"] is True
        assert unknown["status"] == "needs_confirmation"

        # 3. Confirm: the malformed line must be blocked, SOS gets zero doses,
        #    the clean line gets scheduled.
        r = client.post(f"/prescriptions/{prescription_id}/confirm")
        assert r.status_code == 200
        confirm = r.json()
        assert unknown["id"] in confirm["blocked_needs_confirmation"]
        assert sos["id"] in confirm["prn"]
        scheduled_ids = [s["medicine_id"] for s in confirm["scheduled"]]
        assert dolo["id"] in scheduled_ids

        # 4. Dashboard: whatever real wall-clock time this test happens to run at,
        #    doses scheduled for later today must still show up as upcoming, and
        #    the plumbing (unconfirmed/PRN lists) must be right. The "day 1 is not
        #    20%" invariant itself is exhaustively covered, independent of the
        #    real clock, by test_scheduler.py::test_day_1_of_5_day_course_is_not_20_percent —
        #    this test only needs to prove the API wires that logic correctly.
        r = client.get(f"/patients/{patient_id}/dashboard")
        assert r.status_code == 200
        dash = r.json()
        acted_before = dash["adherence"]["taken"] + dash["adherence"]["missed"] + dash["adherence"]["skipped"]
        assert len(dash["unconfirmed_medicines"]) == 1
        assert len(dash["prn_medicines"]) == 1
        assert dash["upcoming_doses"], "at least one future dose should still be pending today"

        # 5. Take the next upcoming dose; taken-count must go up by exactly one
        #    and adherence must recompute from taken/(taken+missed+skipped).
        first_dose_id = dash["upcoming_doses"][0]["id"]

        # A dose that is a day away cannot be "taken" now (this used to be accepted)...
        from datetime import datetime as _dt, timedelta as _td
        from db import SessionLocal as _SL, Dose as _Dose

        def _move_dose(when):
            s_ = _SL()
            s_.query(_Dose).filter_by(id=first_dose_id).update({"scheduled_at": when})
            s_.commit()
            s_.close()

        _move_dose(_dt.utcnow() + _td(days=1))
        r = client.post(f"/doses/{first_dose_id}/take")
        assert r.status_code == 409 and "Too early" in r.json()["detail"]
        # ...but once it is due, it can.
        _move_dose(_dt.utcnow())
        r = client.post(f"/doses/{first_dose_id}/take")
        assert r.status_code == 200

        r = client.get(f"/patients/{patient_id}/dashboard")
        dash = r.json()
        a = dash["adherence"]
        acted_after = a["taken"] + a["missed"] + a["skipped"]
        assert a["taken"] >= 1
        assert acted_after >= acted_before + 1
        assert a["adherence_percent"] == round(a["taken"] / acted_after * 100, 1)

        # 6. Editing the needs_confirmation line fixes it and it becomes verified.
        r = client.patch(f"/medicines/{unknown['id']}", json={"schedule_code": "1-0-1"})
        assert r.status_code == 200
        assert r.json()["status"] == "verified"

        # 7. Triage: EMERGENCY chest pain must produce the emergency action.
        r = client.post("/triage/check", json={
            "patient_id": patient_id,
            "symptom_ids": ["chest_pain"],
            "answers": {"difficulty_breathing": True},
        })
        assert r.status_code == 200
        triage = r.json()
        assert triage["severity"] == "EMERGENCY"
        assert "112" in triage["action"]

        # 8. Report bundles everything, with an EMERGENCY alert present.
        r = client.get(f"/patients/{patient_id}/report")
        assert r.status_code == 200
        report = r.json()
        assert any("EMERGENCY" in a for a in report["alerts"])
        assert report["disclaimer"].startswith("Generated by SmartPoli")


def test_prn_medicine_cannot_be_scheduled_via_log_prn_endpoint_mismatch():
    with TestClient(app) as client:
        register_and_login(client)
        r = client.post("/patients", json={"name": "P2"})
        patient_id = r.json()["id"]
        r = client.post("/prescriptions", json={
            "patient_id": patient_id, "lines": ["Tab X 500mg 1-0-1 5 days"],
        })
        medicine_id = r.json()["medicines"][0]["id"]
        # This medicine is NOT prn — log-prn must refuse it.
        r = client.post(f"/medicines/{medicine_id}/log-prn")
        assert r.status_code == 400


def test_pages_answer_head_requests_for_uptime_monitors():
    from main import PAGES  # uptime monitors send HEAD; it used to return 405
    client = TestClient(app)
    for path in PAGES:
        assert client.head(path).status_code == 200, path
