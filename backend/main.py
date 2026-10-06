"""
SmartPoli — FastAPI app.

Wires together the full journey end to end (CLAUDE.md section 5):

    image  -> OCR plugin  -----\
                                 -> parser -> Medicine[] -> review gate -> confirm -> Dose[] -> dashboard -> report
    textarea (manual) ---------/
                                                     (triage runs independently of the prescription)

Runs with an empty .env: nothing here calls an external API or needs a key.
The manual path never touches the OCR plugin (backend/ocr_plugin.py) —
that only loads (and downloads its ~1.3GB model) the first time someone
actually uploads an image, and any failure there falls back to "use manual
entry" rather than breaking anything else (CLAUDE.md section 5).
"""

import hmac
import io
import json
import logging
import os
import re
from datetime import datetime, timedelta
from typing import Optional

from dotenv import load_dotenv
# .env lives at the project root (one level up from backend/), not inside
# backend/ itself — matches .env.example's location and CLAUDE.md section 13.
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))

from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.gzip import GZipMiddleware
from web_security import SecurityHeadersMiddleware, not_found_handler, rate_limit
from fastapi import FastAPI, HTTPException, Depends, Request, UploadFile, File, Form, Header, BackgroundTasks
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response
from sqlalchemy.orm import Session

from apscheduler.schedulers.background import BackgroundScheduler

from db import (
    init_db, get_db_session, SessionLocal, log_audit, Patient, Prescription, Medicine, Dose, SymptomCheck,
    AuditLog, User, VoiceMessage, PatientSettings, PushSubscription, ReminderLog,
)
from parser import parse_medicine_line, compute_status
from prescription_service import create_prescription_from_lines, confirm_prescription_doses, create_manual_medicine
from scheduler import (
    generate_doses, SchedulingBlocked, compute_adherence, compute_medicine_breakdown,
    mark_taken, mark_missed, mark_skipped, snooze, sweep_missed, sweep_missed_doses, undo_taken,
    take_window_error, TAKE_EARLY_WINDOW,
)
from triage import load_ruleset, evaluate_check, next_question
from triage_service import record_symptom_check
import voice_assistant
import voice_stt
from interactions import load_ruleset as load_interaction_ruleset, check_interactions, normalize_for_interactions
from food_warnings import load_ruleset as load_food_ruleset, check_food_warnings
from ocr_plugin import read_prescription_image, read_image_text_lines, OCRUnavailable
import medicine_scan
from report_pdf import build_report_pdf
from llm_helper import interpret_free_text, is_available as llm_is_available, LLMUnavailable
from i18n import to_plain_language_hi, localized_symptom_label, localized_question_text, localized_action
from schemas import (
    PatientCreate, PatientEdit, ManualMedicine, ShiftMedicine, PrescriptionCreate, MedicineEdit, SkipDose, PrnLog,
    TriageCheckRequest, NextQuestionRequest, ClinicalNoteCreate, FreeTextTriageRequest,
    EmergencyProfileUpdate, VoiceTurnRequest,
    PatientSettingsUpdate, PushSubscribeRequest, PushUnsubscribeRequest, RescheduleDose, RoutineUpdate, TakeDose,
)
from serializers import (
    get_patient_or_404, get_prescription_or_404, get_medicine_or_404, get_dose_or_404,
    serialize_patient, serialize_medicine, serialize_dose,
    gather_report_data, emergency_card_data as _emergency_card_data,
    card_profile, card_photo, save_card_profile,
)
from auth import (
    get_current_user, require_patient_write_access, require_patient_read_access,
    require_medicine_write_access, require_dose_write_access, require_prescription_write_access,
    has_write_access, has_read_access,
)
from nudges import compute_nudges
from clock import patient_now, is_valid_timezone, patient_timezone
from verification import needs_patient_review
from safety import build_safety_center
from ics_export import build_ics
from emergency_tokens import patient_id_for_token, rotate_token
from emergency_page import render_public_card, INACTIVE_CARD_HTML
from health_card_pdf import build_wallet_card_pdf
import auth_router
import caregiver_router
import doctor_router
import whatsapp_router
import whatsapp_bot
import reminders
import regulatory
import routine
import take_guard
import gap_ai
import safety_service
import safety_engine
import webpush_service
import medicine_info
import pair_check
from timeline import friendly_entries

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# The interactive API docs map every endpoint for an attacker, so they are off unless SMARTPOLI_ENABLE_DOCS=1.
_DOCS_ON = os.getenv("SMARTPOLI_ENABLE_DOCS") == "1"
app = FastAPI(title="SmartPoli API", version="1.0.0",
              description="Prescription decoder, adherence scheduler, symptom triage and care report.",
              docs_url="/docs" if _DOCS_ON else None, redoc_url="/redoc" if _DOCS_ON else None,
              openapi_url="/openapi.json" if _DOCS_ON else None)

_default_origins = "http://localhost:3000,http://127.0.0.1:3000,http://localhost:8000,http://127.0.0.1:8000"
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("SMARTPOLI_ALLOWED_ORIGINS", _default_origins).split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "Accept", "Accept-Language"],
)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(GZipMiddleware, minimum_size=600)          # app.js / style.css shrink about 4x on a slow phone connection
app.add_exception_handler(StarletteHTTPException, not_found_handler)

RULESET = load_ruleset()
INTERACTION_RULESET = load_interaction_ruleset()
FOOD_RULESET = load_food_ruleset()

app.include_router(auth_router.router)
app.include_router(caregiver_router.router)
app.include_router(doctor_router.router)
import ml_router  # noqa: E402
app.include_router(ml_router.router)
app.include_router(whatsapp_router.router)

class PublicStaticFiles(StaticFiles):
    """/static serves only the front-end's own assets: an allow-list of extensions, no dotfiles, no source maps,
    no server-side files, and no directory listing. The HTML pages are NOT served from here - they live at clean
    routes (/, /login, /voice, /caregiver, /doctor) - so a /static/*.html request is redirected to its clean URL."""

    ALLOWED = {".js", ".css", ".png", ".svg", ".webmanifest", ".ico", ".jpg", ".jpeg", ".webp", ".woff", ".woff2"}

    async def get_response(self, path, scope):
        name = path.rsplit("/", 1)[-1]
        ext = os.path.splitext(name)[1].lower()
        if name.startswith(".") or ".." in path or ext not in self.ALLOWED:
            raise HTTPException(404, "Not found")
        return await super().get_response(path, scope)


# One explicit route table for the pages: only these URLs return an HTML page.
PAGES = {"/": "index.html", "/login": "login.html", "/voice": "voice.html", "/install": "install.html",
         "/caregiver": "caregiver.html", "/doctor": "doctor.html"}


def _static_version() -> str:
    """Changes whenever the app's files change: the deploy's commit if Render gives us one, else a hash of file times."""
    commit = os.getenv("RENDER_GIT_COMMIT")
    if commit:
        return commit[:10]
    import hashlib
    h = hashlib.sha1()
    for root, _, files in os.walk("static"):
        for f in sorted(files):
            p = os.path.join(root, f)
            h.update(f"{p}:{os.path.getmtime(p):.0f}".encode())
    return h.hexdigest()[:10]


STATIC_VERSION = _static_version()
_ASSET_URL = re.compile(r'(["\'])(/static/[A-Za-z0-9_\-.]+\.(?:js|css))\1')
_page_cache: dict = {}


def _versioned_html(file_name: str) -> str:
    """The page with ?v=<version> on every /static script and stylesheet (cached in memory; files do not change at runtime)."""
    if file_name not in _page_cache:
        text = open(f"static/{file_name}", encoding="utf-8").read()
        _page_cache[file_name] = _ASSET_URL.sub(lambda m: f"{m.group(1)}{m.group(2)}?v={STATIC_VERSION}{m.group(1)}", text)
    return _page_cache[file_name]


def _page_route(file_name: str):
    def serve():
        return HTMLResponse(_versioned_html(file_name), headers={"Cache-Control": "no-cache"})
    return serve


for _path, _file in PAGES.items():
    app.add_api_route(_path, _page_route(_file), methods=["GET"], include_in_schema=False)


@app.get("/static/{name}.html", include_in_schema=False)
def legacy_static_page(name: str, request: Request):
    """Old links (/static/index.html, ...) keep working: redirect to the clean URL, keeping ?query and #hash-less."""
    target = "/" if name == "index" else f"/{name}"
    if target not in PAGES:
        raise HTTPException(404, "Not found")
    q = request.url.query
    return RedirectResponse(target + (f"?{q}" if q else ""), status_code=308)


app.mount("/static", PublicStaticFiles(directory="static"), name="static")


@app.get("/manifest.webmanifest", include_in_schema=False)
def app_manifest():
    """Web app manifest for the whole app (scope "/"): what Chrome and the
    Android Trusted Web Activity read. The older /static/manifest.webmanifest
    belongs to the voice page only."""
    return FileResponse("static/app.webmanifest", media_type="application/manifest+json",
                        headers={"Cache-Control": "no-cache"})


@app.get("/sw-voice.js", include_in_schema=False)
def sw_voice():
    """Voice page's service worker, served from the root so its scope can be /voice."""
    return FileResponse("static/sw-voice.js", media_type="application/javascript",
                        headers={"Cache-Control": "no-cache", "Service-Worker-Allowed": "/voice"})


@app.get("/sw-push.js", include_in_schema=False)
def push_service_worker():
    """Served from the site root so its scope can be "/". Never cached, so a
    fixed worker reaches users immediately."""
    return FileResponse("static/sw-push.js", media_type="application/javascript",
                        headers={"Cache-Control": "no-cache"})


@app.get("/.well-known/assetlinks.json", include_in_schema=False)
def asset_links():
    """Digital Asset Links: proves the Android app (package + signing-key
    fingerprint) and this website belong together, which is what lets the
    Trusted Web Activity open full-screen without a browser address bar.
    Both values come from the environment (docs/ANDROID_TWA.md); until they
    are set this returns an empty list, which verifies nothing - on purpose."""
    package = os.getenv("SMARTPOLI_ANDROID_PACKAGE", "").strip()
    fingerprints = [f.strip().upper() for f in os.getenv("SMARTPOLI_ANDROID_SHA256", "").split(",") if f.strip()]
    if not package or not fingerprints:
        return Response("[]", media_type="application/json")
    return Response(json.dumps([{
        "relation": ["delegate_permission/common.handle_all_urls"],
        "target": {"namespace": "android_app", "package_name": package, "sha256_cert_fingerprints": fingerprints},
    }]), media_type="application/json")


reminder_scheduler = BackgroundScheduler()


_last_external_sweep = [0.0]


def _in_process_sweep_job():
    """The built-in timer is only a fallback: when the external scheduler (cron-job.org) has just run the sweep, skip -
    two sweeps a minute against the remote database is wasted work on a small server."""
    import time as _time
    if _time.time() - _last_external_sweep[0] < 150:
        return
    _reminder_sweep_job()


def _reminder_sweep_job():
    """
    The proactive half of Feature 2's reminders (CLAUDE.md section 9): runs
    on a timer instead of only when a page happens to load. Each tick:
      1. marks doses >2h overdue as missed (judged on each patient's own clock),
         and sends the one-off "missed dose" notification for those;
      2. sends every heads-up / due / follow-up reminder that is owed
         (Web Push + opted-in WhatsApp), each at most once per dose.
    Every step is isolated so one failure cannot stop the others.
    """
    db = SessionLocal()
    try:
        missed = sweep_missed_doses(db)
        if missed:
            logger.info(f"Reminder sweep: auto-marked {len(missed)} overdue dose(s) as missed.")
            try:
                reminders.notify_missed(db, missed)
            except Exception:
                logger.exception("Reminder sweep: missed-dose notifications failed.")
        try:
            sent = reminders.run_reminder_sweep(db)
            if sent:
                logger.info(f"Reminder sweep: sent {sent} reminder message(s).")
        except Exception:
            logger.exception("Reminder sweep: reminder delivery failed.")
    finally:
        db.close()


def _check_secure_config() -> None:
    """Loud warnings (or a refusal to start, with SMARTPOLI_REQUIRE_SECURE_CONFIG=1)
    for settings that are fine on a laptop and unsafe in production."""
    import auth as _auth
    problems = []
    if _auth.JWT_SECRET.startswith("dev-only-insecure"):
        problems.append("SMARTPOLI_JWT_SECRET is not set: sessions are signed with a public development secret.")
    if os.getenv("SMARTPOLI_SKIP_TWILIO_SIGNATURE") == "1":
        problems.append("SMARTPOLI_SKIP_TWILIO_SIGNATURE=1: WhatsApp webhook requests are NOT being verified.")
    for p in problems:
        logger.warning("SECURITY CONFIG: %s", p)
    if problems and os.getenv("SMARTPOLI_REQUIRE_SECURE_CONFIG") == "1":
        raise RuntimeError("Refusing to start with an insecure configuration: " + " ".join(problems))


@app.post("/internal/reminder-sweep", include_in_schema=False)
def internal_reminder_sweep(x_cron_secret: Optional[str] = Header(None)):
    """
    Trigger one reminder sweep from OUTSIDE the process. The in-process timer
    only runs while the server is awake, and a free Render instance sleeps
    after ~15 minutes without traffic - so reminders would silently stop. An
    external scheduler (cron-job.org, a Render Cron Job, GitHub Actions ...)
    calling this every minute both runs the sweep and keeps the instance awake.

    Guarded by SMARTPOLI_CRON_SECRET in the X-Cron-Secret header. When the
    secret is unset or wrong the endpoint answers 404, as if it did not exist.
    Safe to overlap with the in-process timer: every reminder is deduplicated
    in the database.
    """
    secret = os.getenv("SMARTPOLI_CRON_SECRET", "")
    if not secret or not x_cron_secret or not hmac.compare_digest(secret.encode(), x_cron_secret.encode()):
        raise HTTPException(404, "Not Found")
    import time as _time
    _last_external_sweep[0] = _time.time()
    _reminder_sweep_job()
    return {"ok": True}


@app.on_event("startup")
def on_startup():
    init_db()
    if os.getenv("SMARTPOLI_DEMO_SEED") == "1":      # demo copy only: demo accounts + the Digital Twin demo patients
        try:
            import seed as _seed
            import twin_demo_seed
            _seed.seed()
            twin_demo_seed.seed_twin_demo()
        except Exception:
            logging.getLogger("smartpoli").exception("demo seeding failed")
    if os.getenv("SMARTPOLI_AUTO_SEED") == "1":
        # Opt-in only (e.g. Render, where there's no shell on the free plan
        # to run `python seed.py` by hand). seed() itself is idempotent —
        # skips if demo data already exists — so this is safe on every boot.
        from seed import seed
        try:
            seed()
        except Exception as e:
            logger.warning(f"Auto-seed skipped/failed (non-fatal): {e}")
    _check_secure_config()
    if os.getenv("SMARTPOLI_DISABLE_SCHEDULER") == "1":
        # Tests drive the sweeps directly with an injected clock; a live timer
        # sweeping the shared test database would make them flaky.
        logger.info("SmartPoli API up. Background reminder sweep disabled (SMARTPOLI_DISABLE_SCHEDULER=1).")
        return
    reminder_scheduler.add_job(_in_process_sweep_job, "interval", minutes=1, id="reminder_sweep",
                               max_instances=1, coalesce=True, replace_existing=True)
    if not reminder_scheduler.running:
        reminder_scheduler.start()
    logger.info("SmartPoli API up. Manual path only — zero API keys required. Reminder sweep running every minute.")


@app.on_event("shutdown")
def on_shutdown():
    if reminder_scheduler.running:
        reminder_scheduler.shutdown(wait=False)


# ---------------------------------------------------------------- patients
#
# Every patient row now has a real owner (Patient.user_id). Creating one
# requires a logged-in 'patient' account and ties the row to it; reading or
# editing requires either being that owner or (for GET) holding an active
# caregiver/doctor link — auth.require_patient_write_access /
# require_patient_read_access, never a frontend-supplied id alone (Part 18).

@app.post("/patients")
def create_patient(body: PatientCreate, user: User = Depends(get_current_user),
                    db: Session = Depends(get_db_session)):
    if user.role != "patient":
        raise HTTPException(403, "Only a patient account can create a patient profile.")
    patient = Patient(user_id=user.id, name=body.name, age=body.age, sex=body.sex,
                       blood_group=body.blood_group,
                       allergies=body.allergies, emergency_contact=body.emergency_contact)
    db.add(patient)
    db.commit()
    log_audit(db, patient.id, f"patient:{user.id}", "patient_created", body.name)
    return serialize_patient(patient)


@app.get("/patients")
def list_patients(user: User = Depends(get_current_user), db: Session = Depends(get_db_session)):
    """Only the calling patient's own profiles — this used to return every
    patient in the system to anyone (Part 18: 'no unrestricted patient
    APIs'). Caregivers/doctors have their own /caregiver and /doctor
    listings, which are relationship-scoped."""
    if user.role != "patient":
        raise HTTPException(403, "Use /caregiver/patients or /doctor/patients for this role.")
    return [serialize_patient(p) for p in db.query(Patient).filter(Patient.user_id == user.id).all()]


@app.get("/patients/{patient_id}")
def get_patient(patient_id: int, user: User = Depends(require_patient_read_access),
                 db: Session = Depends(get_db_session)):
    return serialize_patient(get_patient_or_404(db, patient_id))


@app.patch("/patients/{patient_id}")
def edit_patient(patient_id: int, body: PatientEdit, user: User = Depends(require_patient_write_access),
                  db: Session = Depends(get_db_session)):
    patient = get_patient_or_404(db, patient_id)
    for field in ("name", "age", "sex", "blood_group", "allergies", "emergency_contact"):
        value = getattr(body, field)
        if value is not None:
            setattr(patient, field, value)
    db.commit()
    log_audit(db, patient.id, f"patient:{user.id}", "patient_updated", "profile edited")
    return serialize_patient(patient)


# ---------------------------------------------------------------- prescriptions (Feature 1)
#
# create_prescription_from_lines lives in prescription_service.py now —
# shared with whatsapp_bot.py, so a prescription line typed, photographed,
# or sent over WhatsApp goes through the exact same parser/confidence path
# (CLAUDE.md section 5/6).
_create_prescription_from_lines = create_prescription_from_lines


@app.post("/prescriptions")
def create_prescription(body: PrescriptionCreate, user: User = Depends(get_current_user),
                         db: Session = Depends(get_db_session)):
    """Manual path: raw lines in, parsed+confidence-scored Medicine rows out. Nothing scheduled yet.
    Only the owning patient may add a prescription to their own record."""
    if not has_write_access(db, user, body.patient_id):
        raise HTTPException(403, "Only the patient can add a prescription to their own record.")
    prescription, medicines = _create_prescription_from_lines(
        db, body.patient_id, body.doctor_name, body.issued_date,
        [(line, None) for line in body.lines], source="manual", actor=f"patient:{user.id}",
    )

    return {
        "prescription_id": prescription.id,
        "status": prescription.status,
        "medicines": [
            {**serialize_medicine(m), "plain_language": parsed["plain_language"], "notes": parsed["notes"]}
            for m, parsed in medicines
        ],
    }


MAX_IMAGE_BYTES = 10 * 1024 * 1024


def _looks_like_image(b: bytes) -> bool:
    """Magic-byte check: the client-supplied filename/content-type is not trusted."""
    return (b[:3] == b"\xff\xd8\xff" or b[:8] == b"\x89PNG\r\n\x1a\n" or b[:2] == b"BM"
            or (b[:4] == b"RIFF" and b[8:12] == b"WEBP") or b[:4] in (b"II*\x00", b"MM\x00*"))


async def _read_image_upload(file: UploadFile) -> bytes:
    """Read an uploaded image with a hard size cap and a magic-byte type check
    (the client-supplied filename and content-type are never trusted)."""
    data = await file.read(MAX_IMAGE_BYTES + 1)
    if not data:
        raise HTTPException(400, "Empty file upload.")
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(413, f"That image is larger than {MAX_IMAGE_BYTES // (1024 * 1024)} MB. "
                                 "Take a smaller photo or use manual entry.")
    if not _looks_like_image(data):
        raise HTTPException(415, "That file is not a supported image (JPEG, PNG, WebP, BMP or TIFF).")
    return data


@app.post("/medicines/scan")
async def scan_medicine(
    patient_id: int = Form(...),
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session),
):
    """
    Camera scan of a medicine strip or box. Returns CANDIDATE names read from
    the packaging for the patient to confirm - nothing is added here, no
    schedule is guessed and the photo is not stored. The patient then adds the
    medicine through the normal prescription path (POST /prescriptions), which
    applies the usual review/confirm gate.
    """
    get_patient_or_404(db, patient_id)
    if not has_write_access(db, user, patient_id):
        raise HTTPException(403, "Only the patient can add a medicine to their own record.")
    image_bytes = await _read_image_upload(file)
    # 1) a vision model reads the box (the main way); 2) the older OCR + name matching is only the backup.
    result = await run_in_threadpool(medicine_scan.identify_with_vision, image_bytes)
    if result is None:
        try:
            lines = await run_in_threadpool(read_image_text_lines, image_bytes)
        except OCRUnavailable:
            lines = []
        result = medicine_scan.candidates_from_lines(lines) if lines else None
    if result is None or not result["candidates"]:
        raise HTTPException(422, "We could not read a medicine from that photo. Enter it manually instead.")
    log_audit(db, patient_id, f"patient:{user.id}", "medicine_scanned",
              f"{len(result['candidates'])} candidate(s) from {len(result['lines_read'])} text line(s)")
    return result


@app.post("/medicines/manual")
def add_medicine_manually(body: ManualMedicine, background: BackgroundTasks, user: User = Depends(get_current_user),
                          db: Session = Depends(get_db_session)):
    """Add a medicine from the popup (typed, or after a scan) and schedule it right away - no confirm step."""
    if not has_write_access(db, user, body.patient_id):
        raise HTTPException(403, "Only the patient can add a medicine to their own record.")
    try:
        out = create_manual_medicine(db, body, f"patient:{user.id}")
    except ValueError as e:
        raise HTTPException(422, str(e))
    _maybe_warm(body.patient_id, background)
    background.add_task(medicine_info.warm_patient, body.patient_id)          # look up what it is for, in the background
    med = out["medicine"]
    first = (db.query(Dose).filter(Dose.medicine_id == med.id, Dose.state == "pending")
             .order_by(Dose.scheduled_at).first())
    # Things the person should hear about right now: timing clashes with their other medicines, and known interactions.
    me = (med.name or med.raw_text or "").strip().lower()
    conflicts = [c for c in safety_service.conflicts_for_patient(db, body.patient_id)["conflicts"]
                 if any(me and me == (n or "").strip().lower() for n in c.get("medicines", []))]
    other_rows = [(m.id, m.name) for m in (db.query(Medicine).join(Prescription)
                                           .filter(Prescription.patient_id == body.patient_id, Medicine.status != "needs_confirmation",
                                                   Medicine.id != med.id).all()) if m.name]
    others = [n for _, n in other_rows]
    id_of = {normalize_for_interactions(n): i for i, n in other_rows}
    mine = normalize_for_interactions(med.name or "")
    # Minor interactions never interrupt: only the ones the table rates MODERATE or CRITICAL.
    interactions = []
    for i in check_interactions(INTERACTION_RULESET, [med.name or ""] + others):
        if mine in (i["drug_a"], i["drug_b"]) and i["severity"] in ("MODERATE", "CRITICAL"):
            ids = [med.id if g == mine else id_of.get(g) for g in (i["drug_a"], i["drug_b"])]
            interactions.append({**i, "medicine_ids": ids})
    return {"prescription_id": out["prescription_id"], "medicine": serialize_medicine(med),
            "doses_scheduled": sum(s["doses_generated"] for s in out["scheduled"]["scheduled"]),
            "first_dose": first.scheduled_at.isoformat() if first else None,
            "conflicts": conflicts, "interactions": interactions}


@app.get("/medicines/{medicine_id}/pair-check")
def medicine_pair_check(medicine_id: int, user: User = Depends(require_medicine_write_access),
                        db: Session = Depends(get_db_session)):
    """Should this medicine be kept apart from the others the patient takes? Sourced (FDA label + AI, graded), cached.
    Called right after a medicine is added; the answer is a list of warnings, each with a suggested later time."""
    medicine = get_medicine_or_404(db, medicine_id)
    now = patient_now(db, medicine.prescription.patient_id)
    return {"warnings": pair_check.warnings_for_new_medicine(db, medicine, now),
            "note": "SmartPoli only knows the combinations its sources cover. No warning is not proof two medicines are safe together."}


@app.get("/prescriptions/{prescription_id}/pair-check")
def prescription_pair_check(prescription_id: int, user: User = Depends(require_prescription_write_access),
                            db: Session = Depends(get_db_session)):
    """Same 'keep these apart' check as for one new medicine, for every medicine of a typed / photographed prescription."""
    prescription = get_prescription_or_404(db, prescription_id)
    now = patient_now(db, prescription.patient_id)
    return {"warnings": pair_check.warnings_for_prescription(db, prescription, now),
            "note": "SmartPoli only knows the combinations its sources cover. No warning is not proof two medicines are safe together."}


@app.get("/patients/{patient_id}/pair-check")
def patient_pair_check(patient_id: int, background: BackgroundTasks, user: User = Depends(require_patient_read_access),
                       db: Session = Depends(get_db_session)):
    """Safety center: every pair among the patient's medicines that should be kept apart, from the cache. Pairs not yet
    looked up are fetched in the background (`pending`) and show up on the next call."""
    get_patient_or_404(db, patient_id)
    out = pair_check.warnings_for_patient(db, patient_id, patient_now(db, patient_id))
    if out["pending"] and gap_ai.is_enabled():
        background.add_task(gap_ai.warm_patient, patient_id)
    out["note"] = "SmartPoli only knows the combinations its sources cover. No warning is not proof two medicines are safe together."
    return out


@app.delete("/medicines/{medicine_id}")
def remove_medicine(medicine_id: int, user: User = Depends(require_medicine_write_access),
                    db: Session = Depends(get_db_session)):
    """Remove a medicine the person no longer takes (or added by mistake). Reminders stop and it disappears from every
    list, but nothing already recorded is erased: taken / missed history stays in the timeline and adherence. Upcoming
    doses are cancelled, not deleted, so past reminder records stay consistent."""
    medicine = get_medicine_or_404(db, medicine_id)
    pid = medicine.prescription.patient_id
    cancelled = 0
    for d in medicine.doses:
        if d.state in ("pending", "snoozed"):
            d.state = "cancelled"
            cancelled += 1
    try:
        fc = json.loads(medicine.field_confidence) if medicine.field_confidence else {}
    except ValueError:
        fc = {}
    fc["removed"] = True
    medicine.field_confidence = json.dumps(fc)
    medicine.status = "needs_confirmation"               # every list and check already skips these
    db.commit()
    log_audit(db, pid, f"patient:{user.id}", "medicine_removed", f"medicine {medicine.id}: {cancelled} upcoming dose(s) cancelled")
    return {"removed": True, "medicine_id": medicine.id, "upcoming_doses_cancelled": cancelled}


@app.get("/patients/{patient_id}/open-warnings")
def patient_open_warnings(patient_id: int, background: BackgroundTasks, user: User = Depends(require_patient_read_access),
                          db: Session = Depends(get_db_session)):
    """Everything that is a REAL problem right now, for the popup shown when the app is opened: medicines that interact
    (rated MODERATE or CRITICAL), pairs a source says must be kept apart while they are scheduled together, and verified
    timing clashes. Minor interactions and anything uncertain never appear. `key` changes when the set of problems does."""
    get_patient_or_404(db, patient_id)
    meds = (db.query(Medicine).join(Prescription, Medicine.prescription_id == Prescription.id)
            .filter(Prescription.patient_id == patient_id, Medicine.status != "needs_confirmation").all())
    ids_by_generic: dict = {}
    for m in meds:
        if m.name:
            ids_by_generic.setdefault(normalize_for_interactions(m.name), []).append(m.id)
    interactions = []
    for i in check_interactions(INTERACTION_RULESET, [m.name for m in meds if m.name]):
        if i["severity"] in ("MODERATE", "CRITICAL"):
            interactions.append({**i, "medicine_ids": [ids_by_generic.get(i["drug_a"], [None])[0], ids_by_generic.get(i["drug_b"], [None])[0]]})
    pairs = pair_check.warnings_for_patient(db, patient_id, patient_now(db, patient_id))
    if pairs["pending"] and gap_ai.is_enabled():
        background.add_task(gap_ai.warm_patient, patient_id)
    timing = safety_service.conflicts_for_patient(db, patient_id)["conflicts"]
    import hashlib
    sig = json.dumps([sorted((i["drug_a"], i["drug_b"]) for i in interactions),
                      sorted(tuple(sorted(w["medicines"])) for w in pairs["warnings"]),
                      sorted(tuple(sorted(c["medicines"])) for c in timing)], default=str)
    return {"interactions": interactions, "pair_warnings": pairs["warnings"], "timing": timing, "pending": pairs["pending"],
            "has_problems": bool(interactions or pairs["warnings"] or timing),
            "key": hashlib.sha1(sig.encode()).hexdigest()[:16]}


@app.post("/medicines/{medicine_id}/shift")
def shift_medicine_times(medicine_id: int, body: ShiftMedicine, user: User = Depends(require_medicine_write_access),
                         db: Session = Depends(get_db_session)):
    """Accept a 'keep these apart' suggestion: move every UPCOMING, untouched dose of this medicine later by the same
    number of minutes. Past and already-acted doses are never touched; no dose may move onto another day."""
    medicine = get_medicine_or_404(db, medicine_id)
    pid = medicine.prescription.patient_id
    now = patient_now(db, pid)
    doses = [d for d in medicine.doses if d.state in ("pending", "snoozed") and d.scheduled_at >= now]
    if not doses:
        raise HTTPException(409, "There are no upcoming doses of this medicine to move.")
    delta = timedelta(minutes=body.minutes)
    if any((d.scheduled_at + delta).date() != d.scheduled_at.date() for d in doses):
        raise HTTPException(422, "That would move a dose onto another day. Choose the time yourself instead.")
    for d in doses:
        d.scheduled_at = d.scheduled_at + delta
    try:
        times = json.loads(medicine.times) if medicine.times else []
        medicine.times = json.dumps(sorted({(datetime.strptime(t, "%H:%M") + delta).strftime("%H:%M") for t in times
                                            if (datetime.strptime(t, "%H:%M") + delta).day == datetime.strptime(t, "%H:%M").day}))
    except ValueError:
        pass
    db.commit()
    log_audit(db, pid, f"patient:{user.id}", "medicine_time_changed", f"medicine {medicine.id} moved later by {body.minutes} min (keep-apart suggestion)")
    return {"medicine_id": medicine.id, "doses_moved": len(doses), "times": json.loads(medicine.times or "[]")}


@app.get("/patients/{patient_id}/medicine-info")
def patient_medicine_info(patient_id: int, background: BackgroundTasks, user: User = Depends(require_patient_read_access),
                          db: Session = Depends(get_db_session)):
    """What each of the patient's medicines is usually for (plain words, from public drug labels), plus the conditions
    that suggests. Answers come from the cache; anything missing is looked up in the background (`pending`)."""
    get_patient_or_404(db, patient_id)
    out = medicine_info.summarize_patient(db, patient_id, allow_network=False)
    if out["pending"] and out["enabled"]:
        background.add_task(medicine_info.warm_patient, patient_id)
    return out


@app.post("/prescriptions/from-image")
async def create_prescription_from_image(
    patient_id: int = Form(...),
    doctor_name: Optional[str] = Form(None),
    issued_date: Optional[str] = Form(None),
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session),
):
    """
    Image path: photo -> OpenCV preprocessing -> per-line TrOCR -> the SAME
    parser.py the manual path uses. No LLM extraction step and no API key —
    OCR only turns pixels into text lines; the deterministic parser (already
    the source of truth for the manual path) does everything after that,
    so there is exactly one extraction/confidence/scheduling path regardless
    of source (CLAUDE.md section 5/6).

    If the OCR plugin can't run at all (model failed to load, bad image),
    this returns a clear 503 telling the caller to use manual entry —
    it never leaves the manual path degraded.
    """
    get_patient_or_404(db, patient_id)
    if not has_write_access(db, user, patient_id):
        raise HTTPException(403, "Only the patient can add a prescription to their own record.")

    image_bytes = await _read_image_upload(file)

    try:
        # run_in_threadpool: OCR is synchronous, CPU-bound work (OpenCV +
        # Tesseract) — running it inline on this async def's event loop
        # would block every other request on this single-worker instance,
        # including Render's own health check, for the whole duration.
        # A real phone photo (megapixels) did exactly that in production:
        # the health check timed out and Render killed the instance mid
        # request, returning 502 to the caller.
        ocr_lines = await run_in_threadpool(read_prescription_image, image_bytes)
    except OCRUnavailable as e:
        raise HTTPException(
            503,
            f"OCR is unavailable right now ({e}). Use manual prescription entry instead — "
            "it covers every feature the image path does.",
        )

    if not ocr_lines:
        raise HTTPException(
            422,
            "No readable text lines were found in that image. Try a clearer photo, "
            "or use manual entry.",
        )

    prescription, medicines = _create_prescription_from_lines(
        db, patient_id, doctor_name, issued_date,
        [(line["text"], line["confidence"]) for line in ocr_lines], source="ocr", actor=f"patient:{user.id}",
    )

    return {
        "prescription_id": prescription.id,
        "status": prescription.status,
        "ocr_lines_found": len(ocr_lines),
        "medicines": [
            {**serialize_medicine(m), "plain_language": parsed["plain_language"], "notes": parsed["notes"]}
            for m, parsed in medicines
        ],
    }


@app.patch("/medicines/{medicine_id}")
def edit_medicine(medicine_id: int, body: MedicineEdit, user: User = Depends(require_medicine_write_access),
                   db: Session = Depends(get_db_session)):
    """
    Review & confirm screen (CLAUDE.md section 8): a human corrects a
    needs_confirmation medicine. Re-derives the schedule from any corrected
    schedule_code and marks it verified — a human just looked at it.
    """
    medicine = get_medicine_or_404(db, medicine_id)
    before = {f: getattr(medicine, f) for f in
              ("name", "dose_amount", "dose_unit", "schedule_code", "food", "duration_days", "is_prn", "status")}

    from shorthand import decode_schedule

    if body.name is not None:
        medicine.name = body.name
        medicine.normalized_name = body.name
    if body.dose_amount is not None:
        medicine.dose_amount = body.dose_amount
    if body.dose_unit is not None:
        medicine.dose_unit = body.dose_unit

    if body.schedule_code is not None:
        schedule = decode_schedule(body.schedule_code)
        medicine.schedule_code = schedule["scheduleCode"] or body.schedule_code
        medicine.slots = json.dumps(schedule["slots"])
        medicine.times = json.dumps(schedule["times"])
        medicine.is_prn = schedule["prn"]
        if schedule["durationDays"] is not None or schedule["ongoing"]:
            medicine.duration_days = schedule["durationDays"]
        medicine.plain_language_hi = to_plain_language_hi(schedule)

    if body.food is not None:
        medicine.food = body.food
    if body.duration_days is not None:
        medicine.duration_days = body.duration_days
    if body.ongoing:
        medicine.duration_days = None

    # A human has now reviewed and corrected this line — it is verified.
    medicine.confidence = 1.0
    medicine.field_confidence = json.dumps({"name": 1.0, "schedule": 1.0, "source": "human_confirmed"})
    medicine.status = "verified"

    db.commit()
    changes = {f: [before[f], getattr(medicine, f)] for f in before if before[f] != getattr(medicine, f)}
    log_audit(db, medicine.prescription.patient_id, f"patient:{user.id}", "medicine_confirmed",
              json.dumps({"medicine_id": medicine.id, "raw_text": medicine.raw_text, "corrections": changes,
                          "confirmed_at": datetime.utcnow().isoformat() + "Z"}, default=str))
    return serialize_medicine(medicine)


@app.post("/prescriptions/{prescription_id}/confirm")
def confirm_prescription(prescription_id: int, background: BackgroundTasks,
                          user: User = Depends(require_prescription_write_access),
                          db: Session = Depends(get_db_session)):
    """
    Generate Dose rows for every medicine that is safe to schedule.

    needs_confirmation medicines are silently skipped here (they stay in the
    prescription, unscheduled, for later editing) — but if anything upstream
    ever tries to force one through generate_doses() directly, that raises
    SchedulingBlocked. The gate lives in the scheduler, not in this endpoint.
    """
    prescription = get_prescription_or_404(db, prescription_id)
    result = confirm_prescription_doses(db, prescription, f"patient:{user.id}")
    _maybe_warm(prescription.patient_id, background)
    return result


# ---------------------------------------------------------------- doses & dashboard (Feature 2)

def _guard_payload(issue: dict, can_override: bool) -> dict:
    """409 body for a refused tap. `detail` stays a plain sentence (older clients just show it); the rest lets the
    app say why and offer 'I already took it' only where that is allowed."""
    return {"detail": issue["message"], "code": issue["code"], "can_override": can_override,
            "earliest": issue.get("earliest"), "source": issue.get("source"), "source_label": issue.get("source_label"),
            "quote": issue.get("quote"), "medicine": issue.get("medicine"), "other_medicine": issue.get("other_medicine")}


def _slot_of(medicine: Medicine, dose: Dose) -> Optional[str]:
    """'morning' / 'afternoon' / ... for a dose, read from the medicine's own slot list (None for exact-time medicines)."""
    try:
        slots, times = json.loads(medicine.slots or "[]"), json.loads(medicine.times or "[]")
    except ValueError:
        return None
    hhmm = dose.scheduled_at.strftime("%H:%M")
    return slots[times.index(hhmm)] if hhmm in times and len(slots) == len(times) else None


_last_warm: dict = {}


def _maybe_warm(patient_id: int, background: BackgroundTasks) -> None:
    """Fill the gap cache in the background (at most once every 6 hours per patient) so taps are answered from it."""
    import time
    if gap_ai.is_enabled() and time.time() - _last_warm.get(patient_id, 0) > 6 * 3600:
        _last_warm[patient_id] = time.time()
        background.add_task(gap_ai.warm_patient, patient_id)


def _dose_patient_id(dose: Dose) -> int:
    return dose.medicine.prescription.patient_id


@app.post("/doses/{dose_id}/take")
def take_dose(dose_id: int, body: Optional[TakeDose] = None, user: User = Depends(require_dose_write_access),
              db: Session = Depends(get_db_session)):
    """
    Record a dose as taken, at the time it was ACTUALLY taken (the patient's
    own clock - a late dose keeps its real time, which the timing-safety
    engine then uses for spacing). Idempotent and race-safe: the state flip
    is one conditional UPDATE, so a double tap or a retried request leaves the
    first recorded time untouched and never double-counts.
    """
    dose = get_dose_or_404(db, dose_id)
    patient_id = _dose_patient_id(dose)
    previous_state, scheduled = dose.state, dose.scheduled_at
    now = patient_now(db, patient_id)
    if dose.state == "taken":
        return serialize_dose(dose)                      # idempotent: a repeat tap changes nothing
    issues = take_guard.evaluate(db, dose, now)
    hard = [i for i in issues if i["hard"]]
    if hard:
        return JSONResponse(status_code=409, content=_guard_payload(hard[0], can_override=False))
    soft = [i for i in issues if not i["hard"]]
    overridden = bool(soft and body and body.override)
    if soft and not overridden:
        return JSONResponse(status_code=409, content=_guard_payload(soft[0], can_override=True))
    changed = (db.query(Dose).filter(Dose.id == dose_id, Dose.state != "taken")
               .update({"state": "taken", "acted_at": now}, synchronize_session=False))
    db.commit()
    db.refresh(dose)
    if changed:
        late = (now - scheduled) > timedelta(minutes=15)
        if overridden:
            log_audit(db, patient_id, f"patient:{user.id}", "dose_taken_override",
                      json.dumps({"dose_id": dose.id, "warnings": [{k: i.get(k) for k in ("code", "source", "other_medicine", "earliest")}
                                                                      for i in soft]}))
        log_audit(db, patient_id, f"patient:{user.id}",
                  "dose_taken_late" if late else "dose_taken",
                  f"dose {dose.id} was {previous_state}; scheduled {scheduled.isoformat()}, "
                  f"recorded {now.isoformat()}")
    return serialize_dose(dose)


@app.post("/doses/{dose_id}/undo")
def undo_dose(dose_id: int, user: User = Depends(require_dose_write_access), db: Session = Depends(get_db_session)):
    dose = get_dose_or_404(db, dose_id)
    try:
        undo_taken(dose, now=patient_now(db, _dose_patient_id(dose)))
    except ValueError as e:
        raise HTTPException(400, str(e))
    db.commit()
    log_audit(db, dose.medicine.prescription.patient_id, f"patient:{user.id}", "dose_taken_undone", f"dose {dose.id}")
    return serialize_dose(dose)


@app.post("/doses/{dose_id}/miss")
def miss_dose(dose_id: int, user: User = Depends(require_dose_write_access), db: Session = Depends(get_db_session)):
    dose = get_dose_or_404(db, dose_id)
    if dose.state == "taken":
        raise HTTPException(409, "This dose is recorded as taken. Undo it first if that was a mistake.")
    if dose.state != "missed":
        mark_missed(dose, acted_at=patient_now(db, _dose_patient_id(dose)))
        db.commit()
        log_audit(db, dose.medicine.prescription.patient_id, f"patient:{user.id}", "dose_missed", f"dose {dose.id}")
    return serialize_dose(dose)


@app.post("/doses/{dose_id}/skip")
def skip_dose(dose_id: int, body: SkipDose, user: User = Depends(require_dose_write_access),
               db: Session = Depends(get_db_session)):
    dose = get_dose_or_404(db, dose_id)
    if dose.state == "taken":
        raise HTTPException(409, "This dose is recorded as taken. Undo it first if that was a mistake.")
    try:
        mark_skipped(dose, body.reason, acted_at=patient_now(db, _dose_patient_id(dose)))
    except ValueError as e:
        raise HTTPException(400, str(e))
    db.commit()
    log_audit(db, dose.medicine.prescription.patient_id, f"patient:{user.id}", "dose_skipped",
              f"dose {dose.id}: {body.reason}")
    return serialize_dose(dose)


@app.post("/doses/{dose_id}/snooze")
def snooze_dose(dose_id: int, user: User = Depends(require_dose_write_access), db: Session = Depends(get_db_session)):
    dose = get_dose_or_404(db, dose_id)
    if dose.state in ("taken", "skipped", "missed"):
        raise HTTPException(409, f"A dose that is already {dose.state} cannot be snoozed.")
    if dose.scheduled_at > patient_now(db, _dose_patient_id(dose)) + TAKE_EARLY_WINDOW:
        raise HTTPException(409, "This dose is not due yet, so there is nothing to snooze.")
    snooze(dose)
    # The new time needs its own 'due' / follow-up reminders; the heads-up stays as sent.
    db.query(ReminderLog).filter(ReminderLog.dose_id == dose.id, ReminderLog.kind.in_(("due", "followup"))).delete(
        synchronize_session=False)
    db.commit()
    return serialize_dose(dose)


# ---------------------------------------------------------------- timing safety (missed-dose guidance, conflicts)
#
# All answers come from safety_engine.py + safety_rules.json (quoted US FDA
# label sentences). Nothing here is generated by an LLM, and a medicine or
# pair without a verified rule is reported as "not verified", never as safe.

@app.get("/doses/{dose_id}/missed-guidance")
def dose_missed_guidance(dose_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db_session)):
    dose = get_dose_or_404(db, dose_id)
    if not has_read_access(db, user, _dose_patient_id(dose)):
        raise HTTPException(403, "You do not have access to this patient's records.")
    if dose.state not in ("missed", "pending", "snoozed"):
        raise HTTPException(409, f"This dose is {dose.state}; there is nothing to advise.")
    return safety_service.missed_guidance_for_dose(db, dose)


@app.get("/patients/{patient_id}/schedule-conflicts")
def schedule_conflicts(patient_id: int, user: User = Depends(require_patient_read_access),
                       db: Session = Depends(get_db_session)):
    get_patient_or_404(db, patient_id)
    return safety_service.conflicts_for_patient(db, patient_id)


@app.post("/doses/{dose_id}/reschedule")
def reschedule_dose(dose_id: int, body: RescheduleDose, user: User = Depends(require_dose_write_access),
                    db: Session = Depends(get_db_session)):
    """
    Accept a spacing proposal. The server re-derives the answer from the
    verified rules and refuses any time the engine would not itself suggest
    (earlier than allowed, at/after the medicine's next dose, or breaking
    another rule) - the client's time is never trusted. The original time is
    kept in the audit trail; dose, frequency and course length are unchanged.
    """
    dose = get_dose_or_404(db, dose_id)
    patient_id = _dose_patient_id(dose)
    if dose.state not in ("pending", "snoozed"):
        raise HTTPException(409, f"A dose that is {dose.state} cannot be rescheduled.")
    try:
        target = datetime.fromisoformat(body.to).replace(tzinfo=None)
    except ValueError:
        raise HTTPException(400, "'to' must be an ISO date-time like 2026-10-05T14:00:00.")
    if dose.scheduled_at == target:
        return serialize_dose(dose)  # idempotent retry of an already-applied move
    medicines, now = safety_service.engine_inputs(db, patient_id)
    ok, reason = safety_engine.validate_reschedule(medicines, safety_service.rules(), now, dose_id, target)
    if not ok:
        raise HTTPException(422, reason)
    original = dose.scheduled_at
    dose.scheduled_at = target
    db.query(ReminderLog).filter(ReminderLog.dose_id == dose.id).delete(synchronize_session=False)
    db.commit()
    safety_service.record_decision(
        db, patient_id, f"patient:{user.id}", "dose_rescheduled_by_rule",
        {"dose_id": dose.id, "from": original.isoformat(), "to": target.isoformat()},
        {"action": "reschedule", "status": "accepted_by_patient"})
    return serialize_dose(dose)


# ---------------------------------------------------------------- settings, push, regulatory

@app.get("/patients/{patient_id}/settings")
def get_patient_settings(patient_id: int, user: User = Depends(require_patient_read_access),
                         db: Session = Depends(get_db_session)):
    get_patient_or_404(db, patient_id)
    row = db.query(PatientSettings).filter(PatientSettings.patient_id == patient_id).first()
    return {"timezone": patient_timezone(db, patient_id),
            "timezone_is_set": bool(row and row.timezone),
            "reminder_lead_minutes": row.reminder_lead_minutes if row and row.reminder_lead_minutes is not None
            else reminders.DEFAULT_LEAD_MINUTES}


@app.put("/patients/{patient_id}/settings")
def put_patient_settings(patient_id: int, body: PatientSettingsUpdate,
                         user: User = Depends(require_patient_write_access), db: Session = Depends(get_db_session)):
    get_patient_or_404(db, patient_id)
    if body.timezone is not None and not is_valid_timezone(body.timezone):
        raise HTTPException(422, "Unknown timezone. Use an IANA name such as Asia/Kolkata.")
    row = db.query(PatientSettings).filter(PatientSettings.patient_id == patient_id).first()
    if row is None:
        row = PatientSettings(patient_id=patient_id)
        db.add(row)
    changed = []
    if body.timezone is not None and body.timezone != row.timezone:
        changed.append(f"timezone {row.timezone} -> {body.timezone}")
        row.timezone = body.timezone
    if body.reminder_lead_minutes is not None and body.reminder_lead_minutes != row.reminder_lead_minutes:
        changed.append(f"reminder lead -> {body.reminder_lead_minutes} min")
        row.reminder_lead_minutes = body.reminder_lead_minutes
    db.commit()
    if changed:
        log_audit(db, patient_id, f"patient:{user.id}", "settings_updated", "; ".join(changed))
    return get_patient_settings(patient_id, user, db)


@app.get("/patients/{patient_id}/routine")
def get_patient_routine(patient_id: int, user: User = Depends(require_patient_read_access),
                        db: Session = Depends(get_db_session)):
    """When this patient's morning / afternoon / evening / night / bedtime doses happen, and their reminder options."""
    get_patient_or_404(db, patient_id)
    return routine.get_routine(db, patient_id)


@app.put("/patients/{patient_id}/routine")
def put_patient_routine(patient_id: int, body: RoutineUpdate, user: User = Depends(require_patient_write_access),
                        db: Session = Depends(get_db_session)):
    get_patient_or_404(db, patient_id)
    times = {s_: getattr(body, s_) for s_ in ("morning", "afternoon", "evening", "night", "bedtime")}
    problem = routine.validate_times(times)
    if problem:
        raise HTTPException(422, problem)
    from db import PatientRoutine
    row = routine.routine_row(db, patient_id)
    if row is None:
        row = PatientRoutine(patient_id=patient_id)
        db.add(row)
    before = routine.get_routine(db, patient_id)
    for s_, v in times.items():
        setattr(row, s_, v)
    row.notify_soon, row.notify_followup = body.notify_soon, body.notify_followup
    db.commit()
    after = routine.get_routine(db, patient_id)
    if before["times"] != after["times"] or before["notify_soon"] != after["notify_soon"] \
            or before["notify_followup"] != after["notify_followup"]:
        log_audit(db, patient_id, f"patient:{user.id}", "routine_updated",
                  json.dumps({"times": after["times"], "notify_soon": after["notify_soon"],
                              "notify_followup": after["notify_followup"]}))
    return after


@app.post("/patients/{patient_id}/routine/apply")
def apply_patient_routine(patient_id: int, user: User = Depends(require_patient_write_access),
                          db: Session = Depends(get_db_session)):
    """Move this patient's current, untouched FUTURE doses to their saved routine. Doses already taken,
    skipped, missed, or moved for a spacing rule are left alone. Safe to repeat."""
    get_patient_or_404(db, patient_id)
    if routine.routine_row(db, patient_id) is None:
        raise HTTPException(409, "Save your daily routine first.")
    result = routine.move_pending_doses(db, patient_id, patient_now(db, patient_id))
    if result["doses_moved"] or result["medicines_changed"]:
        log_audit(db, patient_id, f"patient:{user.id}", "routine_applied", json.dumps(result))
        try:
            reminders.notify_new_conflicts(db, patient_id)      # new times can change which medicines are close together
        except Exception:
            logger.exception("conflict notice after routine apply failed")
    return result


@app.get("/push/public-key")
def push_public_key():
    """The VAPID public key a browser needs to subscribe (public by design)."""
    return {"configured": webpush_service.is_configured(), "public_key": webpush_service.public_key()}


@app.post("/patients/{patient_id}/push/subscribe")
def push_subscribe(patient_id: int, body: PushSubscribeRequest, user: User = Depends(require_patient_write_access),
                   db: Session = Depends(get_db_session)):
    get_patient_or_404(db, patient_id)
    if not body.endpoint.startswith("https://"):
        raise HTTPException(422, "Push endpoints must be https.")
    sub = db.query(PushSubscription).filter(PushSubscription.endpoint == body.endpoint).first()
    if sub is None:
        sub = PushSubscription(endpoint=body.endpoint, patient_id=patient_id, user_id=user.id,
                               p256dh=body.keys.p256dh, auth=body.keys.auth)
        db.add(sub)
    else:  # same device re-subscribing (or a new account on a shared device): re-point, never duplicate
        sub.patient_id, sub.user_id, sub.p256dh, sub.auth = patient_id, user.id, body.keys.p256dh, body.keys.auth
    db.commit()
    return {"subscribed": True,
            "count": db.query(PushSubscription).filter(PushSubscription.patient_id == patient_id).count()}


@app.post("/patients/{patient_id}/push/unsubscribe")
def push_unsubscribe(patient_id: int, body: PushUnsubscribeRequest, user: User = Depends(require_patient_write_access),
                     db: Session = Depends(get_db_session)):
    db.query(PushSubscription).filter(PushSubscription.endpoint == body.endpoint,
                                      PushSubscription.patient_id == patient_id).delete(synchronize_session=False)
    db.commit()
    return {"subscribed": False}


_last_push_test: dict = {}


@app.post("/patients/{patient_id}/push/test")
def push_test(patient_id: int, user: User = Depends(require_patient_write_access),
              db: Session = Depends(get_db_session)):
    """Send one test notification to this patient's registered devices - the
    way to check, on the real phone, that reminders actually arrive."""
    import time
    get_patient_or_404(db, patient_id)
    if not webpush_service.is_configured():
        raise HTTPException(503, "Push reminders are not configured on this server.")
    if time.time() - _last_push_test.get(patient_id, 0) < 10:
        raise HTTPException(429, "Please wait a few seconds between tests.")
    _last_push_test[patient_id] = time.time()
    subs = db.query(PushSubscription).filter(PushSubscription.patient_id == patient_id).all()
    sent = 0
    for sub in subs:
        status = webpush_service.send(sub.endpoint, sub.p256dh, sub.auth, {
            "title": "SmartPoli test", "body": "Reminders are working on this device.",
            "url": "/", "tag": "smartpoli-test", "kind": "test"})
        if status == "sent":
            sent += 1
        elif status == "gone":
            db.delete(sub)
    db.commit()
    return {"devices": len(subs), "sent": sent}


@app.get("/patients/{patient_id}/regulatory")
def patient_regulatory(patient_id: int, user: User = Depends(require_patient_read_access),
                       db: Session = Depends(get_db_session)):
    """Official-source status for each of the patient's confirmed medicines."""
    get_patient_or_404(db, patient_id)
    meds = (db.query(Medicine).join(Prescription)
            .filter(Prescription.patient_id == patient_id, Medicine.status != "needs_confirmation").all())
    regulatory.prefetch(db, [m.name or m.raw_text for m in meds])
    seen, out = set(), []
    for m in meds:
        key = (m.name or m.raw_text or "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        strength = f"{m.dose_amount}{m.dose_unit}" if m.dose_amount and m.dose_unit else None
        out.append({"medicine_id": m.id, **regulatory.lookup_medicine(db, m.name or m.raw_text, m.raw_text, strength)})
    return {"medicines": out, "disclaimer": regulatory.DISCLAIMER}


@app.get("/regulatory/lookup")
def regulatory_lookup(name: str, strength: Optional[str] = None, user: User = Depends(get_current_user),
                      db: Session = Depends(get_db_session)):
    if not name.strip() or len(name) > 120:
        raise HTTPException(422, "Give a medicine name (up to 120 characters).")
    return regulatory.lookup_medicine(db, name.strip(), None, strength)


@app.post("/medicines/{medicine_id}/log-prn")
def log_prn(medicine_id: int, user: User = Depends(require_medicine_write_access),
            db: Session = Depends(get_db_session)):
    """PRN/SOS medicines never get Dose rows — logging a use is an AuditLog entry."""
    medicine = get_medicine_or_404(db, medicine_id)
    if not medicine.is_prn:
        raise HTTPException(400, "This medicine is not PRN/SOS.")
    log_audit(db, medicine.prescription.patient_id, f"patient:{user.id}", "prn_taken",
              f"medicine {medicine.id} ({medicine.name or medicine.raw_text}) at {datetime.utcnow().isoformat()}")
    return {"logged": True, "medicine_id": medicine_id, "at": datetime.utcnow().isoformat()}


@app.get("/patients/{patient_id}/dashboard")
def dashboard(patient_id: int, background: BackgroundTasks, user: User = Depends(require_patient_read_access),
              db: Session = Depends(get_db_session)):
    get_patient_or_404(db, patient_id)
    _maybe_warm(patient_id, background)
    sweep_missed(db)

    prescriptions = db.query(Prescription).filter(Prescription.patient_id == patient_id).all()

    medicines_with_doses = []
    upcoming = []
    now = patient_now(db, patient_id)  # dose times are patient-local wall-clock
    for pres in prescriptions:
        for medicine in pres.medicines:
            medicines_with_doses.append((medicine, medicine.doses))
            for dose in medicine.doses:
                if dose.state in ("pending", "snoozed") and dose.scheduled_at >= now:
                    upcoming.append((dose, medicine))

    upcoming.sort(key=lambda pair: pair[0].scheduled_at)
    all_doses = [d for _, doses in medicines_with_doses for d in doses]
    # Every dose from a day and a half either side of now, whatever its state:
    # the voice page works out "due now", "left today" and "did I take it" in
    # the patient's own timezone from this, with no request of its own.
    recent = sorted(((d, m) for m, doses in medicines_with_doses for d in doses
                     if abs(d.scheduled_at - now) <= timedelta(hours=36)), key=lambda pair: pair[0].scheduled_at)

    # What the patient actually sees: TODAY (in their own timezone), every state, in time order.
    # The 30 generated days stay in the database; they are not a screen.
    today = now.date()
    todays = sorted(((d, m) for m, doses in medicines_with_doses for d in doses if d.scheduled_at.date() == today),
                    key=lambda pair: pair[0].scheduled_at)

    return {
        "patient_id": patient_id,
        "today": today.isoformat(),
        "today_doses": [{**serialize_dose(dose), "medicine_name": medicine.name or medicine.raw_text,
                         "food": medicine.food, "slot": _slot_of(medicine, dose)} for dose, medicine in todays],
        "left_today": sum(1 for dose, _ in todays if dose.state in ("pending", "snoozed")),
        "adherence": compute_adherence(all_doses),
        "per_medicine": compute_medicine_breakdown(medicines_with_doses),
        "upcoming_doses": [
            {**serialize_dose(dose), "medicine_name": medicine.name or medicine.raw_text}
            for dose, medicine in upcoming[:20]
        ],
        "recent_doses": [
            {**serialize_dose(dose), "medicine_name": medicine.name or medicine.raw_text}
            for dose, medicine in recent
        ],
        "prn_medicines": [
            serialize_medicine(m) for pres in prescriptions for m in pres.medicines if m.is_prn
        ],
        "unconfirmed_medicines": [
            serialize_medicine(m) for pres in prescriptions for m in pres.medicines
            if needs_patient_review(m)
        ],
        "interactions": check_interactions(
            INTERACTION_RULESET,
            [m.name for pres in prescriptions for m in pres.medicines if m.status != "needs_confirmation" and m.name],
        ),
        "food_warnings": check_food_warnings(
            FOOD_RULESET,
            [m.name for pres in prescriptions for m in pres.medicines if m.status != "needs_confirmation" and m.name],
        ),
        # Part 4 of the brief — deterministic, derived only from the Dose
        # rows already fetched above. Never invents a dosage or overrides
        # the prescription; see nudges.py.
        "nudges": compute_nudges(medicines_with_doses, now),
    }


@app.get("/patients/{patient_id}/safety-center")
def safety_center(patient_id: int, user: User = Depends(require_patient_read_access),
                   db: Session = Depends(get_db_session)):
    """Part 3 of the brief — one aggregated view over the EXISTING interaction
    and food-warning engines plus a narrow dosage sanity check. See safety.py."""
    get_patient_or_404(db, patient_id)
    prescriptions = db.query(Prescription).filter(Prescription.patient_id == patient_id).all()
    medicines = [m for pres in prescriptions for m in pres.medicines]
    return build_safety_center(INTERACTION_RULESET, FOOD_RULESET, medicines)


@app.get("/patients/{patient_id}/calendar.ics")
def patient_calendar_ics(patient_id: int, user: User = Depends(require_patient_read_access),
                          db: Session = Depends(get_db_session)):
    """Part 8 of the brief — exports the EXISTING schedule (Dose rows already
    generated at confirmation) as a standard .ics feed. No second scheduler."""
    patient = get_patient_or_404(db, patient_id)
    prescriptions = db.query(Prescription).filter(Prescription.patient_id == patient_id).all()
    medicines_with_doses = [(m, m.doses) for pres in prescriptions for m in pres.medicines if not m.is_prn]
    ics_bytes = build_ics(medicines_with_doses, datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"))
    filename = f"smartpoli_schedule_{patient.name.replace(' ', '_')}.ics"
    return Response(
        content=ics_bytes, media_type="text/calendar",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/patients/{patient_id}/interactions")
def patient_interactions(patient_id: int, user: User = Depends(require_patient_read_access),
                          db: Session = Depends(get_db_session)):
    """
    Feature D (optional, CLAUDE.md section 4) — pairwise-checks the
    patient's active medicine names against a local, deterministic
    interaction table. Never a diagnosis or treatment instruction, and
    never blocks scheduling — purely informational.
    """
    get_patient_or_404(db, patient_id)
    prescriptions = db.query(Prescription).filter(Prescription.patient_id == patient_id).all()
    names = [m.name for pres in prescriptions for m in pres.medicines if m.status != "needs_confirmation" and m.name]
    return {"patient_id": patient_id, "interactions": check_interactions(INTERACTION_RULESET, names)}


@app.get("/patients/{patient_id}/food-warnings")
def patient_food_warnings(patient_id: int, user: User = Depends(require_patient_read_access),
                           db: Session = Depends(get_db_session)):
    """Feature E (optional, CLAUDE.md section 4) — specific foods/substances
    to avoid per active medicine. Distinct from the shorthand engine's
    before/after-meals timing field."""
    get_patient_or_404(db, patient_id)
    prescriptions = db.query(Prescription).filter(Prescription.patient_id == patient_id).all()
    names = [m.name for pres in prescriptions for m in pres.medicines if m.status != "needs_confirmation" and m.name]
    return {"patient_id": patient_id, "food_warnings": check_food_warnings(FOOD_RULESET, names)}


# ---------------------------------------------------------------- triage (Feature 3)

@app.get("/triage/symptoms")
def get_symptoms(lang: str = "en"):
    return [
        {"id": sid, "label": localized_symptom_label(s, lang)}
        for sid, s in RULESET["symptoms"].items()
    ]


@app.get("/triage/llm-available")
def triage_llm_available():
    return {"available": llm_is_available()}


@app.post("/triage/interpret-free-text")
def triage_interpret_free_text(body: FreeTextTriageRequest, _rl=Depends(rate_limit("triage-llm", 10, 60))):
    """
    Optional LLM assist: free text -> candidate symptom_ids + answers, for
    the user to review before anything is submitted. Never returns a
    severity — evaluate_check() is the only thing that ever decides that.
    """
    try:
        return interpret_free_text(body.text, RULESET)
    except LLMUnavailable as e:
        raise HTTPException(503, f"Free-text interpretation is unavailable ({e}). Use the symptom picker instead.")


@app.get("/triage/symptoms/{symptom_id}/questions")
def get_symptom_questions(symptom_id: str, lang: str = "en"):
    if symptom_id not in RULESET["symptoms"]:
        raise HTTPException(404, f"Unknown symptom {symptom_id}")
    return [
        {"id": q["id"], "text": localized_question_text(q, lang), "type": q["type"]}
        for q in RULESET["symptoms"][symptom_id]["questions"]
    ]


@app.post("/triage/next-question")
def get_next_question(body: NextQuestionRequest, _rl=Depends(rate_limit("triage", 60, 60))):
    if body.symptom_id not in RULESET["symptoms"]:
        raise HTTPException(404, f"Unknown symptom {body.symptom_id}")
    q = next_question(RULESET, body.symptom_id, body.answers, body.current_severity)
    return {"question": q}


@app.post("/triage/preview")
def triage_preview(body: TriageCheckRequest, lang: str = "en", _rl=Depends(rate_limit("triage", 60, 60))):
    """Same evaluation as /triage/check but never persisted — used by the
    question-by-question UI to decide whether to keep asking (rule 2:
    stop once EMERGENCY) without writing a SymptomCheck row per keystroke."""
    for sid in body.symptom_ids:
        if sid not in RULESET["symptoms"]:
            raise HTTPException(404, f"Unknown symptom {sid}")
    result = evaluate_check(RULESET, body.symptom_ids, body.answers)
    return {**result, "action": localized_action(result["action"], lang)}


@app.post("/triage/check")
def triage_check(body: TriageCheckRequest, user: User = Depends(get_current_user),
                  db: Session = Depends(get_db_session), lang: str = "en"):
    get_patient_or_404(db, body.patient_id)
    if not has_write_access(db, user, body.patient_id):
        raise HTTPException(403, "Only the patient can log a symptom check on their own record.")
    for sid in body.symptom_ids:
        if sid not in RULESET["symptoms"]:
            raise HTTPException(404, f"Unknown symptom {sid}")

    result, check = record_symptom_check(db, body.patient_id, f"patient:{user.id}", RULESET,
                                         body.symptom_ids, body.answers)
    return {**result, "check_id": check.id, "action": localized_action(result["action"], lang)}


# ---------------------------------------------------------------- voice assistant
#
# Groq is only the conversational layer (voice_assistant.py); every action
# goes through voice_tools' validated router and every severity comes from
# triage.py. The voice page (static/voice.html) is the only caller.

def _local_now(client_time: Optional[str]) -> datetime:
    """Dose times are stored as the patient's local wall-clock time, so the
    assistant reasons in the browser's local time, not the server's."""
    try:
        return datetime.fromisoformat(client_time).replace(tzinfo=None) if client_time else datetime.now()
    except ValueError:
        return datetime.now()


@app.delete("/patients/{patient_id}/voice/history")
def delete_voice_history(patient_id: int, user: User = Depends(require_patient_write_access),
                         db: Session = Depends(get_db_session)):
    """The patient can wipe what the voice assistant remembers of their conversations."""
    deleted = db.query(VoiceMessage).filter(VoiceMessage.patient_id == patient_id).delete()
    db.commit()
    log_audit(db, patient_id, f"patient:{user.id}", "voice_history_deleted", f"{deleted} messages")
    return {"deleted": deleted}


@app.get("/voice/available")
def voice_available(user: User = Depends(get_current_user)):
    return {"available": voice_assistant.is_available(), "stt": voice_stt.is_available()}


@app.post("/patients/{patient_id}/voice/transcribe")
async def voice_transcribe(patient_id: int, audio: UploadFile = File(...), lang: str = Form(""),
                           user: User = Depends(require_patient_write_access)):
    """One recorded utterance → text, via Groq Whisper (voice_stt.py)."""
    content_type = (audio.content_type or "").split(";")[0].strip().lower()
    if content_type not in voice_stt.ALLOWED_TYPES:
        raise HTTPException(415, "Send the recording as audio.")
    data = await audio.read(voice_stt.MAX_AUDIO_BYTES + 1)
    if len(data) > voice_stt.MAX_AUDIO_BYTES:
        raise HTTPException(413, "That recording is too long.")
    if not voice_assistant.check_rate_limit(user.id):
        raise HTTPException(429, "Too many voice requests. Wait a moment and try again.")
    try:
        text = voice_stt.transcribe(data, audio.filename or "utterance.wav", lang)
    except voice_stt.TranscriptionUnavailable:
        raise HTTPException(503, "Speech recognition service is not configured.")
    except Exception as e:
        logger.warning("Whisper transcription failed: %s", e)
        raise HTTPException(502, "Couldn't transcribe that — please say it again.")
    return {"text": text}


@app.post("/patients/{patient_id}/voice/turn")
def voice_turn(patient_id: int, body: VoiceTurnRequest, user: User = Depends(require_patient_write_access),
               db: Session = Depends(get_db_session)):
    text = body.text.strip()
    if not text:
        raise HTTPException(422, "Say or type something first.")
    if not voice_assistant.check_rate_limit(user.id):
        raise HTTPException(429, "Too many voice requests. Wait a moment and try again.")
    return voice_assistant.run_turn(db, patient_id, user, text, body.lang,
                                    _local_now(body.client_time),
                                    [m.model_dump() for m in body.history], body.state, body.recheck)


# ---------------------------------------------------------------- clinical notes (doctor view)
#
# Reuses AuditLog rather than a new model — CLAUDE.md/the implementation
# brief both call for exactly one audit trail, not a second table just for
# notes. A note is simply an audit entry with action='clinical_note' whose
# actor is the AUTHENTICATED doctor's real identity — never a client-
# supplied 'actor' string (that was the exact hole Part 8 of the brief
# flags: `actor = "doctor"` in the request body proved nothing).

@app.post("/patients/{patient_id}/notes")
def add_clinical_note(patient_id: int, body: ClinicalNoteCreate, user: User = Depends(get_current_user),
                       db: Session = Depends(get_db_session)):
    if user.role != "doctor":
        raise HTTPException(403, "Only a doctor account can add a clinical note.")
    if not has_read_access(db, user, patient_id):
        raise HTTPException(403, "You are not linked to this patient.")
    get_patient_or_404(db, patient_id)
    log_audit(db, patient_id, f"doctor:{user.id}:{user.name}", "clinical_note", body.note)
    return {"logged": True}


_TIMELINE_LABELS = {
    "patient_created": "Patient profile created",
    "patient_updated": "Patient profile updated",
    "prescription_created": "Prescription added",
    "prescription_confirmed": "Prescription confirmed and scheduled",
    "medicine_confirmed": "A flagged medicine was reviewed and confirmed",
    "dose_taken": "Dose marked as taken",
    "dose_missed": "Dose marked as missed",
    "dose_skipped": "Dose skipped",
    "dose_auto_missed": "Dose automatically marked missed (no response after 2h)",
    "prn_taken": "As-needed medicine logged",
    "triage_check": "Symptom check completed",
    "clinical_note": "Doctor/caregiver note added",
    "demo_seeded": "Demo data seeded",
}


@app.get("/patients/{patient_id}/timeline")
def patient_timeline(patient_id: int, user: User = Depends(require_patient_read_access),
                      db: Session = Depends(get_db_session)):
    """
    Feature 'unified patient timeline' — one chronological view over the
    same AuditLog every other feature already writes to (no second,
    competing history table). Turns SmartPoli's separate features into one
    visible treatment journey, which is the whole point of the product.
    """
    get_patient_or_404(db, patient_id)
    entries = (
        db.query(AuditLog)
        .filter(AuditLog.patient_id == patient_id)
        .order_by(AuditLog.at.desc())
        .limit(200)
        .all()
    )
    return friendly_entries(db, entries)


@app.get("/patients/{patient_id}/notes")
def get_clinical_notes(patient_id: int, user: User = Depends(require_patient_read_access),
                        db: Session = Depends(get_db_session)):
    get_patient_or_404(db, patient_id)
    notes = (
        db.query(AuditLog)
        .filter(AuditLog.patient_id == patient_id, AuditLog.action == "clinical_note")
        .order_by(AuditLog.at.desc())
        .all()
    )
    return [{"id": n.id, "actor": n.actor, "note": n.detail, "at": n.at.isoformat()} for n in notes]


# ---------------------------------------------------------------- report (Feature 4)
#
# gather_report_data lives in serializers.py now — the doctor dashboard
# (doctor_router.py) reuses the exact same function, so there is one report
# shape, not a second one built for the doctor view.

@app.get("/patients/{patient_id}/report")
def patient_report(patient_id: int, user: User = Depends(require_patient_read_access),
                    db: Session = Depends(get_db_session)):
    return gather_report_data(db, patient_id, INTERACTION_RULESET, FOOD_RULESET)


@app.get("/patients/{patient_id}/report/pdf")
def patient_report_pdf(patient_id: int, user: User = Depends(require_patient_read_access),
                        db: Session = Depends(get_db_session)):
    """A real, downloadable PDF — not just the browser's Print to PDF —
    built with ReportLab, SmartPoli's own canonical report renderer."""
    data = gather_report_data(db, patient_id, INTERACTION_RULESET, FOOD_RULESET)
    pdf_bytes = build_report_pdf(data)
    filename = f"smartpoli_report_{data['patient']['name'].replace(' ', '_')}.pdf"
    return Response(
        content=pdf_bytes, media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ---------------------------------------------------------------- emergency card (Feature J, optional)
#
# _emergency_card_data (imported above from serializers.py as
# emergency_card_data) is used two ways: the authenticated JSON endpoint
# below requires patient/caregiver/doctor read access like everything else,
# but /emergency/{token} — what the printed QR code actually opens —
# is DELIBERATELY left with no login requirement. A stranger finding an
# unconscious patient does not have that patient's password; the whole
# point of an emergency card is that it works without one (CLAUDE.md
# section 11, Part 15/16 of the brief). It exposes only name/age/sex/
# blood group/allergies/emergency contact/current medicines — nothing else
# from the record.

@app.get("/patients/{patient_id}/emergency-card")
def get_emergency_card_data(patient_id: int, user: User = Depends(require_patient_read_access),
                             db: Session = Depends(get_db_session)):
    return _emergency_card_data(db, patient_id)


@app.get("/patients/{patient_id}/emergency-card/qr.png")
def emergency_card_qr(patient_id: int, request: Request, user: User = Depends(require_patient_read_access),
                       db: Session = Depends(get_db_session)):
    get_patient_or_404(db, patient_id)
    import qrcode
    url = str(request.base_url).rstrip("/") + _emergency_card_data(db, patient_id)["card_path"]
    img = qrcode.make(url, border=2)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


@app.post("/patients/{patient_id}/emergency-card/revoke")
def revoke_emergency_card(patient_id: int, user: User = Depends(require_patient_write_access),
                          db: Session = Depends(get_db_session)):
    """Lost phone or printed card: the old QR/link stops working immediately."""
    get_patient_or_404(db, patient_id)
    token = rotate_token(db, patient_id)
    log_audit(db, patient_id, f"patient:{user.id}", "emergency_card_revoked", "token rotated")
    return {"card_path": f"/emergency/{token}"}


@app.get("/patients/{patient_id}/emergency-card/profile")
def get_card_profile(patient_id: int, user: User = Depends(require_patient_read_access),
                     db: Session = Depends(get_db_session)):
    get_patient_or_404(db, patient_id)
    return card_profile(db, patient_id)


@app.put("/patients/{patient_id}/emergency-card/profile")
def put_card_profile(patient_id: int, body: EmergencyProfileUpdate,
                     user: User = Depends(require_patient_write_access),
                     db: Session = Depends(get_db_session)):
    get_patient_or_404(db, patient_id)
    save_card_profile(db, patient_id, body.model_dump(exclude_unset=True))
    log_audit(db, patient_id, f"patient:{user.id}", "emergency_profile_updated",
              ",".join(sorted(body.model_dump(exclude_unset=True))))
    return card_profile(db, patient_id)


@app.get("/patients/{patient_id}/emergency-card/photo")
def get_card_photo(patient_id: int, user: User = Depends(require_patient_read_access),
                   db: Session = Depends(get_db_session)):
    photo = card_photo(db, patient_id)
    if not photo:
        raise HTTPException(404, "No photo on this card.")
    return Response(content=photo[0], media_type=photo[1])


@app.get("/patients/{patient_id}/emergency-card/card.pdf")
def get_wallet_card_pdf(patient_id: int, request: Request, user: User = Depends(require_patient_read_access),
                        db: Session = Depends(get_db_session)):
    data = _emergency_card_data(db, patient_id)
    photo = card_photo(db, patient_id) if data["profile"]["share"]["photo"] else None
    pdf = build_wallet_card_pdf(data, str(request.base_url).rstrip("/") + data["card_path"],
                                photo[0] if photo else None)
    return Response(content=pdf, media_type="application/pdf",
                    headers={"Content-Disposition": 'attachment; filename="smartpoli-health-card.pdf"'})


_PUBLIC_HEADERS = {"Referrer-Policy": "no-referrer"}


@app.get("/emergency/{token}", response_class=HTMLResponse, include_in_schema=False)
def emergency_card_page(token: str, db: Session = Depends(get_db_session), _rl=Depends(rate_limit("emergency", 60, 60))):
    """Public, standalone, no-login card — what a QR scan opens. Addressed by
    a random revocable token (emergency_tokens.py), never the patient id;
    shows only what the patient chose to share (emergency_page.py)."""
    patient_id = patient_id_for_token(db, token)
    if patient_id is None:
        return HTMLResponse(content=INACTIVE_CARD_HTML, status_code=404, headers=_PUBLIC_HEADERS)
    return HTMLResponse(content=render_public_card(_emergency_card_data(db, patient_id)),
                        headers=_PUBLIC_HEADERS)


@app.get("/emergency/{token}/photo", include_in_schema=False)
def emergency_card_public_photo(token: str, db: Session = Depends(get_db_session), _rl=Depends(rate_limit("emergency", 60, 60))):
    patient_id = patient_id_for_token(db, token)
    photo = card_photo(db, patient_id) if patient_id is not None else None
    if not photo or not card_profile(db, patient_id)["share"]["photo"]:
        raise HTTPException(404, "Not found.")
    return Response(content=photo[0], media_type=photo[1], headers=_PUBLIC_HEADERS)
