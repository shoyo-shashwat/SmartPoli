"""
Loads the Digital Twin demo patients into the demo copy of the app and links them to the demo doctor.

The data files (backend/twin_demo_data/demo_*.json) hold anonymised public research data and are NOT in the
public repo; the demo copy is deployed from a private repo/branch that includes them. If a file is missing it is
skipped, so the app still starts. Safe to run more than once.
"""
import base64
import gzip
import json
import os
import secrets
from datetime import datetime

from db import SessionLocal, init_db, DoctorLink, GlucoseReading, HeartRateReading, Patient, TwinEvent, TwinProfile, User

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "twin_demo_data")
# Private demo patients (not in the public repo) can arrive three ways, checked in this order:
#   1. a Render "Secret File" named demo_b.json (Render mounts them in /etc/secrets),
#   2. a folder named by SMARTPOLI_TWIN_DEMO_DIR,
#   3. an environment variable SMARTPOLI_TWIN_DEMO_B_GZ_B64 holding the gzip + base64 of demo_b.json (see ml/pack_private_demo.py).
EXTRA_DIRS = [os.getenv("SMARTPOLI_TWIN_DEMO_DIR"), "/etc/secrets"]
DOCTOR_EMAIL = "doctor@smartpoli.demo"


def _demo_files(data_dir):
    """{file name: parsed json} from the repo folder, the extra folders and the packed environment variable."""
    found = {}
    for d in [data_dir] + [x for x in EXTRA_DIRS if x]:
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                if f.startswith("demo_") and f.endswith(".json") and f not in found:
                    found[f] = json.load(open(os.path.join(d, f), encoding="utf8"))
    packed = os.getenv("SMARTPOLI_TWIN_DEMO_B_GZ_B64")
    if packed and "demo_b.json" not in found:
        found["demo_b.json"] = json.loads(gzip.decompress(base64.b64decode(packed)).decode("utf8"))
    return found


def seed_twin_demo(data_dir: str = DATA_DIR, doctor_email: str = DOCTOR_EMAIL) -> list:
    init_db()
    done = []
    files = _demo_files(data_dir)
    if not files:
        return done
    db = SessionLocal()
    try:
        doctor = db.query(User).filter(User.email == doctor_email).first()
        for fname, d in sorted(files.items()):
            name = d["patient"]["name"]
            if db.query(Patient).filter(Patient.name == name).first():
                continue
            p = Patient(name=name, age=d["patient"].get("age"), sex=d["patient"].get("sex"))
            db.add(p); db.commit()
            prof = dict(d["profile"]); prof["replay_start"] = datetime.fromisoformat(prof["replay_start"]) if prof.get("replay_start") else None
            db.add(TwinProfile(patient_id=p.id, **prof))
            db.bulk_save_objects([GlucoseReading(patient_id=p.id, timestamp=datetime.fromisoformat(t), value=v, source="replay") for t, v in d["readings"]])
            db.bulk_save_objects([TwinEvent(patient_id=p.id, timestamp=datetime.fromisoformat(t), kind=k, text=txt, iu=iu) for t, k, txt, iu in d["events"]])
            if d.get("heart_rate"):
                db.bulk_save_objects([HeartRateReading(patient_id=p.id, timestamp=datetime.fromisoformat(t), bpm=b) for t, b in d["heart_rate"]])
            if doctor:
                db.add(DoctorLink(patient_id=p.id, doctor_user_id=doctor.id, code=secrets.token_hex(4).upper(), status="active", accepted_at=datetime.utcnow()))
            db.commit()
            done.append(name)
        return done
    finally:
        db.close()


if __name__ == "__main__":
    print("Loaded:", seed_twin_demo() or "nothing new")
