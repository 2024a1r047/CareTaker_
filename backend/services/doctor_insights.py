"""
Doctor portal insights: everything the uploaded reports say about a patient, merged with the
database, plus a rule-based clinical summary and a fallback Q&A for the AI Case Chat.

Read-only: nothing is written to the database. Parsed reports are cached in memory per
(report id, file size), so a newly uploaded report is picked up automatically.
"""
import hashlib
import re
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from services.report_parser import parse_report_text

_PARSE_CACHE: Dict[tuple, Dict[str, Any]] = {}

# Lab rows saved earlier by the old extractor that are clearly not tests (addresses, IDs...)
_JUNK_DB_METRIC = re.compile(
    r"(road|marg|nagar|jaipur|uhid|age ?/ ?sex|height ?/ ?weight|referred|reported|ethnicity|exacerbations|"
    r"borg|fasting sample|spirometry|e\. exertional|confirm\.|walk test|distance walked|mmrc|cat score|sex)", re.I)

_PLAUSIBLE = {"Hemoglobin": (3, 25), "HbA1c": (3, 20), "Fasting Glucose": (20, 800), "eGFR": (2, 200),
              "Creatinine": (0.1, 20), "LDL Cholesterol": (10, 500), "Total Cholesterol": (50, 700)}

# Most useful trends first in the chart buttons
_PRIORITY = ["HbA1c", "Fasting Glucose", "Postprandial Glucose", "eGFR", "Urine Microalbumin", "Creatinine",
             "Hemoglobin", "LDL Cholesterol", "Total Cholesterol", "Triglycerides", "HDL Cholesterol",
             "Vitamin D", "Vitamin B12", "Uric Acid", "Potassium", "Sodium", "CRP", "WBC"]


def _d(x) -> Optional[str]:
    if x is None:
        return None
    if isinstance(x, datetime):
        return x.date().isoformat()
    if isinstance(x, date):
        return x.isoformat()
    return str(x)[:10]


def _fmt_date(x) -> str:
    if not x:
        return "date unknown"
    if isinstance(x, str):
        try:
            x = date.fromisoformat(x[:10])
        except ValueError:
            return x
    return x.strftime("%d %b %Y")


def _g(v) -> str:
    return f"{v:g}" if isinstance(v, (int, float)) else str(v)


def _parsed_reports(recipient_id: int, db) -> List[Dict[str, Any]]:
    from tables.medical_reports import MedicalReport
    from utils.summarizer import extract_text_from_bytes
    reports = (db.query(MedicalReport).filter(MedicalReport.care_recipient_id == recipient_id)
               .order_by(MedicalReport.id).all())
    out, seen_text = [], set()
    for r in reports:
        data = bytes(r.data) if r.data is not None else b""
        key = (r.id, len(data))
        if key not in _PARSE_CACHE:
            try:
                text = extract_text_from_bytes(data, r.mime_type) if data else (r.analysis_summary or "")
            except Exception:
                text = r.analysis_summary or ""
            fallback = r.report_date.date() if isinstance(r.report_date, datetime) else r.report_date
            parsed = parse_report_text(text or "", r.filename or "", fallback)
            parsed["_hash"] = hashlib.md5((text or "").encode("utf-8", "ignore")).hexdigest()
            _PARSE_CACHE[key] = parsed
        p = dict(_PARSE_CACHE[key])
        p.update({"id": r.id, "filename": r.filename, "uploaded_at": r.uploaded_at,
                  "processing_status": getattr(r, "processing_status", None),
                  "duplicate": p["_hash"] in seen_text})
        seen_text.add(p["_hash"])
        out.append(p)
    return out


def build_insights(recipient_id: int, db, recipient=None) -> Dict[str, Any]:
    from tables.medical_conditions import LabValue
    reports = _parsed_reports(recipient_id, db)
    unique = [r for r in reports if not r["duplicate"]]
    parsed_report_ids = {r["id"] for r in reports if r["labs"]}

    # ── Labs: reports + earlier DB rows (junk and misreads filtered out) ──
    series: Dict[str, List[Dict[str, Any]]] = {}

    def add_point(name, d, value, unit, low, high, abnormal, source):
        if value is None:
            return
        lim = _PLAUSIBLE.get(name)
        if lim and not (lim[0] <= value <= lim[1]):
            return
        pts = series.setdefault(name, [])
        if any(p["date"] == d and abs(p["value"] - value) < 1e-6 for p in pts):
            return
        pts.append({"date": d, "value": value, "unit": unit, "is_abnormal": bool(abnormal),
                    "reference_low": low, "reference_high": high, "source": source})

    for r in unique:
        for l in r["labs"]:
            add_point(l["name"], _d(l["date"] or r["report_date"]), l["value"], l["unit"], l["low"], l["high"],
                      l["status"] in ("high", "low"), r["title"])
    for lv in db.query(LabValue).filter(LabValue.care_recipient_id == recipient_id).all():
        if lv.report_id in parsed_report_ids or not lv.metric_name or _JUNK_DB_METRIC.search(lv.metric_name):
            continue
        add_point(lv.metric_name, _d(lv.recorded_date), lv.normalized_value if lv.normalized_value is not None else lv.metric_value,
                  lv.normalized_unit or lv.unit, lv.reference_range_low, lv.reference_range_high, lv.is_abnormal, "lab record")
    for pts in series.values():
        pts.sort(key=lambda p: p["date"] or "")
    order = sorted(series.keys(), key=lambda n: (_PRIORITY.index(n) if n in _PRIORITY else 100,
                                                  -len(series[n]), n))
    lab_trends = {n: series[n] for n in order}

    # ── Vitals from reports ──
    report_vitals = []
    for r in unique:
        for v in r["vitals"]:
            row = {k: v.get(k) for k in ("heart_rate", "systolic_bp", "diastolic_bp", "oxygen_saturation",
                                         "temperature", "respiratory_rate", "height", "weight", "bmi")}
            row.update({"recorded_at": _d(v["date"] or r["report_date"]), "sleep_score": None,
                        "source": "report", "label": v.get("label"), "report": r["title"]})
            report_vitals.append(row)

    # ── Allergies / diagnoses / notes ──
    allergies, seen_a = [], set()
    for r in unique:
        for a in r["allergies"]:
            k = a["allergen"].lower()
            if k in seen_a:
                continue
            seen_a.add(k)
            sev = "severe" if re.search(r"bronchospasm|anaphyla|swelling|angioedema|breath", a["reaction"], re.I) else "moderate"
            allergies.append({"allergen": a["allergen"], "type": "drug", "reaction": a["reaction"] or None,
                              "severity": sev, "source": f"{r['title']} ({_fmt_date(r['report_date'])})"})
    diagnoses = []
    for r in unique:
        for dx in r["diagnoses"]:
            if not any(dx.lower()[:25] == x["name"].lower()[:25] for x in diagnoses):
                diagnoses.append({"name": dx, "date": _d(r["report_date"]), "source": r["title"]})

    report_list = [{
        "id": r["id"], "filename": r["filename"], "title": r["title"],
        "report_date": _d(r["report_date"]), "uploaded_at": _d(r["uploaded_at"]),
        "processing_status": r["processing_status"] or "completed", "lab_count": len(r["labs"]),
        "has_vitals": bool(r["vitals"]), "duplicate": r["duplicate"],
    } for r in reports]
    report_list.sort(key=lambda x: x["report_date"] or "", reverse=True)

    return {"reports": unique, "report_list": report_list, "lab_trends": lab_trends,
            "report_vitals": report_vitals, "allergies": allergies, "diagnoses": diagnoses}


# ─────────────────────────────────────────────────────────────────────────────
# Clinical summary (rule-based, doctor-facing)
# ─────────────────────────────────────────────────────────────────────────────
def _trend_text(name, pts, better="down") -> Optional[str]:
    if len(pts) < 2:
        return None
    first, last = pts[0], pts[-1]
    seq = " → ".join(f"{_g(p['value'])}" for p in pts)
    unit = last.get("unit") or ""
    diff = last["value"] - first["value"]
    base = abs(first["value"]) or 1.0
    if abs(diff) / base < 0.08:
        word = "stable"                      # under ~8% change: not a real trend
    else:
        improving = (diff < 0) if better == "down" else (diff > 0)
        word = "improving" if improving else "worsening"
    status = " (still abnormal)" if last["is_abnormal"] else " (now within range)"
    return f"{name}: {seq} {unit} [{_fmt_date(first['date'])} → {_fmt_date(last['date'])}] — {word}{status}"


def build_clinical_summary(info: Dict[str, Any], conditions: List[dict], meds: List[dict],
                           allergies: List[dict], vitals: List[dict], ins: Dict[str, Any]) -> str:
    L: List[str] = []
    name = info.get("name") or "Patient"
    sex = (info.get("gender") or "").lower()
    head = f"{name}, {info.get('age') or '?'} y {sex[:1].upper() if sex else ''}"
    if info.get("blood_group"):
        head += f", blood group {info['blood_group']}"
    head += f". Risk score {round(info.get('risk_score') or 0)} ({info.get('risk_category') or 'see header'})."
    n_rep = len(ins["reports"])
    if n_rep:
        dates = sorted(_d(r["report_date"]) for r in ins["reports"] if r["report_date"])
        head += f" Based on {n_rep} uploaded report{'s' if n_rep != 1 else ''}" + (f" ({_fmt_date(dates[0])} – {_fmt_date(dates[-1])})." if dates else ".")
    L.append(head)

    labs = ins["lab_trends"]

    def latest(n):
        pts = labs.get(n) or []
        return pts[-1] if pts else None

    # Problem list
    L.append("## Problem list")
    probs = []
    for d in ins["diagnoses"]:
        # one line per disease (discharge summary and PFT both describe the COPD)
        if "copd" in d["name"].lower() and any("copd" in p.lower() for p in probs):
            continue
        probs.append(d["name"])
    for c in conditions:
        if not any(c["disease_name"].lower().split()[0] in p.lower() for p in probs):
            probs.append(f"{c['disease_name']} ({c.get('severity') or 'severity n/a'})")
    eg, ma = latest("eGFR"), latest("Urine Microalbumin")
    if eg and ma and eg["value"] < 90 and ma["value"] > 30:
        stage = "G2" if eg["value"] >= 60 else "G3a" if eg["value"] >= 45 else "G3b+"
        probs.append(f"Early diabetic kidney disease pattern — eGFR {_g(eg['value'])}, albuminuria {_g(ma['value'])} mg/g (CKD {stage}-A2)")
    vd = latest("Vitamin D")
    if vd and vd["value"] < 20:
        probs.append(f"Vitamin D deficiency ({_g(vd['value'])} ng/mL)")
    for p in probs or ["No structured diagnoses found in the uploaded reports."]:
        L.append(f"- {p}")

    # Allergies
    L.append("## Allergies")
    if allergies:
        for a in allergies:
            L.append(f"- {a['allergen']}" + (f" — {a['reaction']}" if a.get("reaction") else "") + (f" ({a['source']})" if a.get("source") else ""))
    else:
        L.append("- No known allergies recorded.")

    # Key lab trends
    L.append("## Key lab trends")
    better_up = {"eGFR", "Hemoglobin", "HDL Cholesterol", "Vitamin D", "Vitamin B12"}
    trend_lines = [t for n in labs if (t := _trend_text(n, labs[n], "up" if n in better_up else "down"))]
    for t in (trend_lines[:9] or ["- Only single readings available so far (no trend yet)."]):
        L.append(t if t.startswith("- ") else f"- {t}")

    # Latest abnormal results
    latest_date = max((p["date"] for pts in labs.values() for p in pts if p["date"]), default=None)
    abn = []
    for n, pts in labs.items():
        p = pts[-1]
        if p["is_abnormal"] and p["date"] == latest_date:
            rng = f"{_g(p['reference_low'])}–{_g(p['reference_high'])}" if p["reference_low"] is not None and p["reference_high"] is not None \
                else (f"> {_g(p['reference_low'])}" if p["reference_low"] is not None else (f"< {_g(p['reference_high'])}" if p["reference_high"] is not None else ""))
            direction = "HIGH" if (p["reference_high"] is not None and p["value"] > p["reference_high"]) else "LOW"
            abn.append(f"{n} {_g(p['value'])} {p['unit'] or ''} {direction}" + (f" (ref {rng})" if rng else ""))
    if abn:
        L.append(f"## Abnormal on latest panel ({_fmt_date(latest_date)})")
        for a in abn:
            L.append(f"- {a}")

    # Vitals
    if vitals:
        L.append("## Vitals")
        for v in vitals[-4:]:
            parts = []
            if v.get("systolic_bp"): parts.append(f"BP {v['systolic_bp']}/{v['diastolic_bp']}")
            if v.get("heart_rate"): parts.append(f"HR {v['heart_rate']}")
            if v.get("respiratory_rate"): parts.append(f"RR {v['respiratory_rate']}")
            if v.get("oxygen_saturation"): parts.append(f"SpO2 {v['oxygen_saturation']}%")
            if v.get("temperature"): parts.append(f"Temp {_g(v['temperature'])} °F")
            if v.get("bmi"): parts.append(f"BMI {_g(v['bmi'])}")
            if parts:
                L.append(f"- {_fmt_date(v.get('recorded_at'))}{' · ' + v['label'] if v.get('label') else ''}: " + ", ".join(parts))
        desat = next((r.get("exertional_desaturation") for r in ins["reports"] if r.get("exertional_desaturation")), None)
        if desat:
            L.append(f"- Exertional desaturation to {desat}% on 6-minute walk.")

    # Lung function
    lf = [n for n in ("FEV1/FVC (post-BD)", "FEV1 (post-BD)", "6-Minute Walk Distance", "CAT score") if n in labs]
    if lf:
        L.append("## Lung function")
        bits = []
        for n in lf:
            p = labs[n][-1]
            extra = ""
            for r in ins["reports"]:
                for l in r["labs"]:
                    if l["name"] == n and l.get("pct_predicted"):
                        extra = f" ({_g(l['pct_predicted'])}% predicted)"
            bits.append(f"{n} {_g(p['value'])} {p['unit'] or ''}{extra}".strip())
        L.append("- " + "; ".join(bits))

    # Medications
    if meds:
        L.append(f"## Current medications ({len(meds)})")
        L.append("- " + "; ".join(f"{m['medicine_name']} {m.get('dosage') or ''} ({m.get('frequency') or m.get('schedule_time') or ''})".replace(" ()", "") for m in meds))

    # Points for review (rule-based, for the doctor to decide)
    flags = []
    names = " ".join(a["allergen"].lower() for a in allergies)
    if "nsaid" in names:
        flags.append("NSAID allergy (bronchospasm): avoid NSAIDs, including for knee/joint pain.")
    if "penicillin" in names:
        flags.append("Penicillin allergy: use non-beta-lactam antibiotics.")
    if "sulfonamide" in names:
        flags.append("Sulfonamide allergy: avoid sulfa drugs.")
    hb = latest("HbA1c")
    if hb and hb["value"] >= 7.0:
        flags.append(f"HbA1c {_g(hb['value'])}% above target — review diabetes therapy.")
    b12 = latest("Vitamin B12")
    if b12 and b12["value"] < 300 and any("metformin" in (m["medicine_name"] or "").lower() for m in meds):
        flags.append(f"B12 low-normal ({_g(b12['value'])} pg/mL) on long-term Metformin — consider supplementation.")
    if vd and vd["value"] < 20:
        flags.append("Vitamin D deficiency — consider replacement.")
    if eg and ma and ma["value"] > 30:
        flags.append("Persistent albuminuria — continue ARB/ACE-i, monitor creatinine and potassium.")
    if any("prednisolone" in (m["medicine_name"] or "").lower() for m in meds):
        flags.append("Prednisolone listed as active — the discharge summary says it was a short course (check if it should be stopped).")
    if flags:
        L.append("## Points for review")
        for f in flags:
            L.append(f"- {f}")

    # Report conclusions
    notes = []
    for r in sorted(ins["reports"], key=lambda r: _d(r["report_date"]) or "", reverse=True):
        for key, label in (("interpretation", "Interpretation"), ("impression", "Impression"),
                           ("recommendations", "Recommendations"), ("follow_up", "Follow-up")):
            if r["notes"].get(key):
                txt = r["notes"][key]
                if len(txt) > 700:                       # cut at the end of a sentence, not mid-word
                    cut = txt[:700].rfind(". ")
                    txt = txt[:cut + 1] if cut > 200 else txt[:700] + "…"
                notes.append(f"- {r['title']} ({_fmt_date(r['report_date'])}) — {label}: {txt}")
    if notes:
        L.append("## From the reports")
        L.extend(notes[:6])

    L.append("Generated automatically from the uploaded reports and records — for clinical review, not a diagnosis.")
    return "\n".join(L)


# ─────────────────────────────────────────────────────────────────────────────
# AI Case Chat fallback (used when the Gemini key is not available)
# ─────────────────────────────────────────────────────────────────────────────
def answer_question(question: str, context: Dict[str, Any]) -> str:
    q = (question or "").lower()
    info = context.get("general_info", {})
    name = info.get("name", "The patient")
    labs = context.get("lab_trends", {}) or {}
    out = []

    # Plain-word topics -> lab names ("how is her kidney?" -> eGFR, Creatinine, Microalbumin)
    topics = {"sugar": ["glucose", "hba1c"], "diabet": ["glucose", "hba1c"],
              "kidney": ["egfr", "creatinine", "microalbumin"], "renal": ["egfr", "creatinine", "microalbumin"],
              "cholesterol": ["cholesterol", "triglycerides"], "lipid": ["cholesterol", "triglycerides"],
              "anaemia": ["hemoglobin"], "anemia": ["hemoglobin"], "lung": ["fev1", "fvc", "6-minute"],
              "copd": ["fev1", "fvc", "6-minute"], "liver": ["alt", "ast", "bilirubin", "albumin"],
              "thyroid": ["tsh"], "vitamin": ["vitamin"]}
    wanted = [k for t, ks in topics.items() if t in q for k in ks]
    hits = [n for n in labs
            if n.lower() in q or n.lower().split(" (")[0] in q or any(k in n.lower() for k in wanted)]
    for n in hits[:6]:
        pts = labs[n]
        seq = "; ".join(f"{_g(p['value'])} {p.get('unit') or ''} ({_fmt_date(p['date'])}{', abnormal' if p['is_abnormal'] else ''})" for p in pts)
        p = pts[-1]
        rng = ""
        if p.get("reference_low") is not None or p.get("reference_high") is not None:
            rng = f" Reference: {_g(p.get('reference_low')) if p.get('reference_low') is not None else ''}–{_g(p.get('reference_high')) if p.get('reference_high') is not None else ''}."
        out.append(f"**{n}**: {seq}.{rng}")
    if "allerg" in q:
        al = context.get("allergies") or []
        out.append("**Allergies**: " + ("; ".join(f"{a['allergen']} ({a.get('reaction') or 'reaction n/a'})" for a in al) if al else "none recorded."))
    if any(w in q for w in ("medic", "drug", "prescri", "tablet", "dose")):
        meds = context.get("medications") or []
        out.append("**Current medications**: " + ("; ".join(f"{m['medicine_name']} {m.get('dosage') or ''} — {m.get('frequency') or ''}" for m in meds) or "none recorded."))
    if any(w in q for w in ("vital", "bp", "blood pressure", "pulse", "heart rate", "spo2", "oxygen", "temperature")):
        vs = context.get("vitals") or []
        if vs:
            out.append("**Vitals**: " + "; ".join(
                f"{_fmt_date(v.get('recorded_at'))}{' (' + v['label'] + ')' if v.get('label') else ''}: "
                + ", ".join(x for x in [f"BP {v['systolic_bp']}/{v['diastolic_bp']}" if v.get("systolic_bp") else "",
                                        f"HR {v['heart_rate']}" if v.get("heart_rate") else "",
                                        f"SpO2 {v['oxygen_saturation']}%" if v.get("oxygen_saturation") else "",
                                        f"Temp {_g(v['temperature'])}°F" if v.get("temperature") else ""] if x)
                for v in vs[-5:]))
        else:
            out.append("**Vitals**: none recorded.")
    if any(w in q for w in ("diagnos", "condition", "problem", "disease")):
        conds = context.get("conditions") or []
        out.append("**Conditions**: " + "; ".join(c["disease_name"] for c in conds))
    summary = context.get("clinical_summary") or ""
    if any(w in q for w in ("summary", "overview", "about", "history", "tell me", "status", "how is", "review", "risk", "plan", "recommend")) or not out:
        sec = []
        keep = False
        for line in summary.splitlines():
            if line.startswith("## "):
                keep = line[3:] in ("Problem list", "Points for review", "Key lab trends", "Allergies")
                if keep:
                    sec.append(f"**{line[3:]}**")
                continue
            if keep:
                sec.append(line)
        out.append("\n".join(sec) if sec else summary[:1200])
    out.append("_(Answered from the uploaded reports and records. Add a valid Gemini API key for free-text AI reasoning.)_")
    return f"Here is what {name}'s records show:\n\n" + "\n\n".join(out)
