"""
SmartPoli — Doctor linking + doctor dashboard (Part 1 of the brief).

Same real-relationship pattern as caregiver_router.py: a doctor never sees
a patient by knowing their id — a DoctorLink must be active first. A
doctor's corrections to a medicine are never silent: every correction
writes a MedicineCorrection row (original value, corrected value, doctor
id, reason) alongside updating the live Medicine row, and clinical notes
carry the doctor's real identity, not a client-supplied 'actor' string.
"""

import json as _json
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from db import get_db_session, log_audit, User, DoctorLink, MedicineCorrection, PrescriptionTemplate
from auth import get_current_user, require_patient_write_access, require_doctor_role, has_read_access, gen_link_code
from schemas import LinkRedeem, MedicineCorrectionCreate, DoctorPrescriptionCreate, PrescriptionTemplateCreate
from serializers import get_patient_or_404, get_medicine_or_404, serialize_patient, serialize_medicine, gather_report_data
from doctor_intelligence import compute_priority, compute_since_last_visit, compute_doctor_brief
from prescription_service import create_prescription_from_lines
from interactions import load_ruleset as load_interaction_ruleset
from food_warnings import load_ruleset as load_food_ruleset

router = APIRouter(tags=["doctor"])

INTERACTION_RULESET = load_interaction_ruleset()
FOOD_RULESET = load_food_ruleset()

_CORRECTABLE_FIELDS = {"name", "dose_amount", "dose_unit", "food", "duration_days", "schedule_code"}


# ---------------------------------------------------------------- patient side: manage invites

@router.post("/patients/{patient_id}/doctor-links")
def create_doctor_link(patient_id: int, user: User = Depends(require_patient_write_access),
                        db: Session = Depends(get_db_session)):
    get_patient_or_404(db, patient_id)
    code = gen_link_code()
    while db.query(DoctorLink).filter(DoctorLink.code == code).first():
        code = gen_link_code()
    link = DoctorLink(patient_id=patient_id, code=code, status="pending")
    db.add(link)
    db.commit()
    log_audit(db, patient_id, f"patient:{user.id}", "doctor_link_created", f"invite issued (link {link.id})")
    return {"id": link.id, "code": link.code, "status": link.status}


@router.get("/patients/{patient_id}/doctor-links")
def list_doctor_links(patient_id: int, user: User = Depends(require_patient_write_access),
                       db: Session = Depends(get_db_session)):
    links = (
        db.query(DoctorLink)
        .filter(DoctorLink.patient_id == patient_id)
        .order_by(DoctorLink.created_at.desc())
        .all()
    )
    out = []
    for link in links:
        doctor_name = None
        if link.doctor_user_id:
            doc = db.query(User).filter(User.id == link.doctor_user_id).first()
            doctor_name = doc.name if doc else None
        out.append({
            "id": link.id, "code": link.code, "status": link.status, "doctor_name": doctor_name,
            "created_at": link.created_at.isoformat(),
            "accepted_at": link.accepted_at.isoformat() if link.accepted_at else None,
        })
    return out


@router.post("/doctor-links/{link_id}/revoke")
def revoke_doctor_link(link_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db_session)):
    link = db.query(DoctorLink).filter(DoctorLink.id == link_id).first()
    if not link:
        raise HTTPException(404, f"No doctor link {link_id}")
    patient = get_patient_or_404(db, link.patient_id)
    if not (user.role == "patient" and patient.user_id == user.id):
        raise HTTPException(403, "Only the patient can revoke a doctor's access.")
    link.status = "revoked"
    db.commit()
    log_audit(db, link.patient_id, f"patient:{user.id}", "doctor_link_revoked", f"link {link.id}")
    return {"id": link.id, "status": link.status}


# ---------------------------------------------------------------- doctor side

@router.post("/doctor/link/redeem")
def redeem_doctor_link(body: LinkRedeem, user: User = Depends(require_doctor_role),
                        db: Session = Depends(get_db_session)):
    code = body.code.strip().upper()
    link = db.query(DoctorLink).filter(DoctorLink.code == code).first()
    if not link:
        raise HTTPException(404, "Invalid invite code.")
    if link.status == "revoked":
        raise HTTPException(409, "This invite has been revoked.")
    if link.status == "active" and link.doctor_user_id != user.id:
        raise HTTPException(409, "This invite has already been used by another doctor.")

    link.doctor_user_id = user.id
    link.status = "active"
    link.accepted_at = datetime.utcnow()
    db.commit()
    log_audit(db, link.patient_id, f"doctor:{user.id}", "doctor_link_accepted", f"link {link.id}")

    patient = get_patient_or_404(db, link.patient_id)
    return {"linked": True, "patient": serialize_patient(patient)}


@router.get("/doctor/patients")
def doctor_patients(user: User = Depends(require_doctor_role), db: Session = Depends(get_db_session)):
    links = db.query(DoctorLink).filter(
        DoctorLink.doctor_user_id == user.id, DoctorLink.status == "active",
    ).all()
    patients = [
        {**serialize_patient(get_patient_or_404(db, link.patient_id)),
         "priority": compute_priority(db, link.patient_id)}
        for link in links
    ]
    order = {"emergency": 0, "high": 1, "medium": 2, "routine": 3}
    patients.sort(key=lambda p: order[p["priority"]["level"]])
    return patients


@router.get("/doctor/patients/{patient_id}")
def doctor_patient_detail(patient_id: int, user: User = Depends(require_doctor_role),
                           db: Session = Depends(get_db_session)):
    """Everything SmartPoli already generated for this patient — prescription,
    OCR/extraction confidence, schedule, adherence, missed doses, symptoms,
    triage history, notes — reusing gather_report_data rather than a second
    query set built just for the doctor view. brief/since_last_visit add the
    pre-consultation summary and change-since-last-note comparison on top."""
    if not has_read_access(db, user, patient_id):
        raise HTTPException(403, "You are not linked to this patient.")
    report = gather_report_data(db, patient_id, INTERACTION_RULESET, FOOD_RULESET)
    from db import TwinProfile
    tp = db.query(TwinProfile).filter(TwinProfile.patient_id == patient_id).first()
    report["data_source"] = tp.data_source if tp else None          # set for research-data demo patients: they have sugar, meals and doses but no symptom checks
    report["brief"] = compute_doctor_brief(db, patient_id)
    report["since_last_visit"] = compute_since_last_visit(db, patient_id, user.id)
    return report


@router.post("/doctor/patients/{patient_id}/prescriptions")
def doctor_write_prescription(patient_id: int, body: DoctorPrescriptionCreate,
                               user: User = Depends(require_doctor_role), db: Session = Depends(get_db_session)):
    """A doctor writing a prescription directly, not just correcting one the
    patient already typed in. Goes through the exact same parser and
    confidence gate as the manual path (CLAUDE.md: one parser, one gate,
    regardless of who typed the line) — a doctor's shorthand can still come
    out needs_confirmation if it's ambiguous. Left as 'draft': the PATIENT
    still confirms it before any doses are scheduled (require_write_access
    stays patient-only everywhere — a doctor never silently schedules
    something onto someone else's phone)."""
    if not has_read_access(db, user, patient_id):
        raise HTTPException(403, "You are not linked to this patient.")
    prescription, medicines = create_prescription_from_lines(
        db, patient_id, doctor_name=user.name, issued_date=datetime.utcnow().date().isoformat(),
        lines_with_confidence=[(line, None) for line in body.lines],
        source="manual", actor=f"doctor:{user.id}:{user.name}",
    )
    return {
        "prescription_id": prescription.id,
        "status": prescription.status,
        "medicines": [serialize_medicine(m) for m, _parsed in medicines],
    }


@router.get("/doctor/templates")
def list_prescription_templates(user: User = Depends(require_doctor_role), db: Session = Depends(get_db_session)):
    templates = (
        db.query(PrescriptionTemplate)
        .filter(PrescriptionTemplate.doctor_user_id == user.id)
        .order_by(PrescriptionTemplate.created_at.desc())
        .all()
    )
    return [
        {"id": t.id, "label": t.label, "lines": _json.loads(t.lines), "created_at": t.created_at.isoformat()}
        for t in templates
    ]


@router.post("/doctor/templates")
def create_prescription_template(body: PrescriptionTemplateCreate, user: User = Depends(require_doctor_role),
                                  db: Session = Depends(get_db_session)):
    if not body.label.strip() or not any(line.strip() for line in body.lines):
        raise HTTPException(400, "A template needs a label and at least one non-empty line.")
    template = PrescriptionTemplate(
        doctor_user_id=user.id, label=body.label.strip(),
        lines=_json.dumps([line for line in body.lines if line.strip()]),
    )
    db.add(template)
    db.commit()
    return {"id": template.id, "label": template.label, "lines": _json.loads(template.lines),
            "created_at": template.created_at.isoformat()}


@router.delete("/doctor/templates/{template_id}")
def delete_prescription_template(template_id: int, user: User = Depends(require_doctor_role),
                                  db: Session = Depends(get_db_session)):
    template = db.query(PrescriptionTemplate).filter(PrescriptionTemplate.id == template_id).first()
    if not template:
        raise HTTPException(404, f"No template {template_id}")
    if template.doctor_user_id != user.id:
        raise HTTPException(403, "You can only delete your own templates.")
    db.delete(template)
    db.commit()
    return {"deleted": True}


@router.post("/doctor/medicines/{medicine_id}/correction")
def doctor_correct_medicine(medicine_id: int, body: MedicineCorrectionCreate,
                             user: User = Depends(require_doctor_role), db: Session = Depends(get_db_session)):
    """
    Doctor corrections are never silent (Part 7 of the brief): this writes
    a MedicineCorrection row with the original and corrected value before
    touching the live Medicine row. raw_text (the original prescription
    line) is never overwritten — only the parsed fields are.
    """
    medicine = get_medicine_or_404(db, medicine_id)
    patient_id = medicine.prescription.patient_id
    if not has_read_access(db, user, patient_id):
        raise HTTPException(403, "You are not linked to this patient.")
    if body.field not in _CORRECTABLE_FIELDS:
        raise HTTPException(400, f"field must be one of: {', '.join(sorted(_CORRECTABLE_FIELDS))}")

    original_value = str(getattr(medicine, body.field))
    warning = None

    if body.field == "schedule_code":
        from shorthand import decode_schedule
        schedule = decode_schedule(body.corrected_value)
        medicine.schedule_code = schedule["scheduleCode"] or body.corrected_value
        medicine.slots = _json.dumps(schedule["slots"])
        medicine.times = _json.dumps(schedule["times"])
        medicine.is_prn = schedule["prn"]
        if schedule["durationDays"] is not None or schedule["ongoing"]:
            medicine.duration_days = schedule["durationDays"]
        if medicine.prescription.status == "confirmed":
            warning = ("This prescription was already confirmed and scheduled. Existing dose rows were "
                       "NOT regenerated — ask the patient to reconfirm if the new schedule should take effect.")
    elif body.field == "duration_days":
        medicine.duration_days = int(body.corrected_value) if body.corrected_value.strip() else None
    else:
        setattr(medicine, body.field, body.corrected_value)

    medicine.status = "verified"
    medicine.confidence = 1.0

    correction = MedicineCorrection(
        medicine_id=medicine.id, doctor_user_id=user.id, field=body.field,
        original_value=original_value, corrected_value=body.corrected_value, reason=body.reason,
    )
    db.add(correction)
    db.commit()

    log_audit(db, patient_id, f"doctor:{user.id}:{user.name}", "medicine_corrected",
              f"medicine {medicine.id} field={body.field}: {original_value!r} -> {body.corrected_value!r} "
              f"({body.reason})")

    return {"medicine": serialize_medicine(medicine), "correction_id": correction.id, "warning": warning}


@router.get("/doctor/medicines/{medicine_id}/corrections")
def get_medicine_corrections(medicine_id: int, user: User = Depends(require_doctor_role),
                              db: Session = Depends(get_db_session)):
    medicine = get_medicine_or_404(db, medicine_id)
    patient_id = medicine.prescription.patient_id
    if not has_read_access(db, user, patient_id):
        raise HTTPException(403, "You are not linked to this patient.")
    corrections = (
        db.query(MedicineCorrection)
        .filter(MedicineCorrection.medicine_id == medicine_id)
        .order_by(MedicineCorrection.created_at.desc())
        .all()
    )
    return [
        {"id": c.id, "field": c.field, "original_value": c.original_value, "corrected_value": c.corrected_value,
         "reason": c.reason, "doctor_user_id": c.doctor_user_id, "created_at": c.created_at.isoformat()}
        for c in corrections
    ]
