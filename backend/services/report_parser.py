"""
Rule-based reader for uploaded medical reports (lab reports, discharge summaries, PFT reports).

Works on the plain text of a report (from utils.summarizer.extract_text_from_bytes) and pulls out:
  - report date (Collected / Reported / Test Date / Discharged, else the date in the file name)
  - lab results with unit, reference range and High/Low flag (section dates such as
    "ADMISSION BLOOD TESTS (22 Aug 2026)" are respected)
  - spirometry rows (FEV1, FVC, FEV1/FVC, PEF) and 6-minute-walk numbers
  - vitals (Temp, Pulse, BP, Resp. rate, SpO2, height / weight) with the admission / discharge date
  - known allergies, final diagnoses
  - doctor's text sections: Interpretation, Impression, Recommendations, Advice, Hospital course

No AI is needed, so this works even when the Gemini key is not available.
Nothing here writes to the database.
"""
import re
from datetime import datetime, date
from typing import Any, Dict, List, Optional, Tuple

_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}
_DATE_RX = r"(\d{1,2})[\s\-/]([A-Za-z]{3})[A-Za-z]*[\s\-/,]+(\d{4})"


def parse_date(s: str) -> Optional[date]:
    m = re.search(_DATE_RX, s or "")
    if m:
        mon = _MONTHS.get(m.group(2).lower()[:3])
        if mon:
            try:
                return date(int(m.group(3)), mon, int(m.group(1)))
            except ValueError:
                pass
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", s or "")
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass
    return None


def _norm_lines(text: str) -> List[str]:
    out = []
    for raw in (text or "").splitlines():
        line = re.sub(r"\s+", " ", raw).strip()
        if line:
            out.append(line)
    return out


# ── Canonical lab names so the same test lines up across reports ──────────────
_ALIASES = {
    "hdl": "HDL Cholesterol", "hdl cholesterol": "HDL Cholesterol",
    "ldl": "LDL Cholesterol", "ldl cholesterol": "LDL Cholesterol",
    "vldl": "VLDL Cholesterol", "vldl cholesterol": "VLDL Cholesterol",
    "total cholesterol": "Total Cholesterol", "triglycerides": "Triglycerides",
    "microalbumin": "Urine Microalbumin", "urine microalbumin": "Urine Microalbumin",
    "bun": "Blood Urea Nitrogen", "blood urea nitrogen": "Blood Urea Nitrogen",
    "hba1c": "HbA1c", "egfr": "eGFR", "hemoglobin": "Hemoglobin", "haemoglobin": "Hemoglobin",
    "fasting glucose": "Fasting Glucose", "postprandial glucose": "Postprandial Glucose",
    "random glucose": "Random Glucose", "vitamin d (25-oh)": "Vitamin D", "vitamin d": "Vitamin D",
    "vitamin b12": "Vitamin B12", "sgpt (alt)": "ALT (SGPT)", "sgot (ast)": "AST (SGOT)",
    "pco2": "pCO2", "po2": "pO2",
}

# Rows that look like "name number" but are not tests
_JUNK_NAME = re.compile(
    r"(road|marg|nagar|jaipur|ph:|page|uhid|lab id|age ?/ ?sex|height ?/ ?weight|referred|reported|collected|"
    r"sample|patient|ethnicity|ward|bed|smoking|session|grade a|dr\.|consultant|fictional|synthetic|"
    r"investigation|interval|exacerbations|borg|^\d)", re.I)

_LAB_RX = re.compile(
    r"^(?P<name>[A-Za-z][A-Za-z0-9 ()/.,+'\-]*?[A-Za-z0-9)])\s+"
    r"(?P<val>[<>]?\d+(?:\.\d+)?)\s*"
    r"(?P<unit>(?:%|[0-9][0-9^.]*(?:/[A-Za-z0-9.^]+)+|[A-Za-zµ][A-Za-z0-9µ^.]*(?:/[A-Za-z0-9.^]+)*))?"
    r"(?:\s+(?P<range>\d+(?:\.\d+)?\s*-\s*\d+(?:\.\d+)?|[<>]\s*\d+(?:\.\d+)?))?"
    r"(?:\s+(?P<flag>High|Low|H|L|Critical))?\s*$")

_HEADINGS = [
    "final diagnosis", "known allergies", "presenting complaints", "examination on admission",
    "investigations", "hospital course", "condition at discharge", "discharge medications",
    "advice", "interpretation", "impression", "recommendations", "symptom scores", "follow-up",
    "chest x-ray", "ecg", "sputum", "abg impression",
]


def _is_heading(line: str) -> bool:
    l = line.lower().strip()
    return any(l == h or l.startswith(h + " ") or l.startswith(h + " &") for h in _HEADINGS) and len(line) < 60


def _section(lines: List[str], start_words: Tuple[str, ...], stop_at_signature=True) -> str:
    """Text of the section whose heading starts with one of start_words."""
    for i, line in enumerate(lines):
        if any(line.lower().startswith(w) for w in start_words) and len(line) < 60:
            buf = []
            for nxt in lines[i + 1:]:
                if _is_heading(nxt) or (stop_at_signature and nxt.startswith("____")) or "SYNTHETIC SAMPLE" in nxt:
                    break
                buf.append(nxt)
            return " ".join(buf).strip()
    return ""


def _float(s: str) -> Optional[float]:
    try:
        return float(str(s).replace(">", "").replace("<", "").strip())
    except (TypeError, ValueError):
        return None


def parse_report_text(text: str, filename: str = "", fallback_date: Optional[date] = None) -> Dict[str, Any]:
    lines = _norm_lines(text)
    full = "\n".join(lines)

    # ── Dates ──
    report_date = None
    for key in (r"Collected", r"Test Date", r"Discharged", r"Reported", r"Date of (?:report|discharge)"):
        m = re.search(key + r"\s*:?\s*" + _DATE_RX, full, re.I)
        if m:
            report_date = parse_date(m.group(0)); break
    if not report_date:
        report_date = parse_date(filename) or fallback_date
    m_adm = re.search(r"Admitted\s*:?\s*" + _DATE_RX, full, re.I)
    m_dis = re.search(r"Discharged\s*:?\s*" + _DATE_RX, full, re.I)
    admitted = parse_date(m_adm.group(0)) if m_adm else None
    discharged = parse_date(m_dis.group(0)) if m_dis else None

    title = ""
    for line in lines[:15]:
        if line.isupper() and 8 < len(line) < 80 and not any(w in line for w in ("HOSPITAL", "LAB (", "PATHOLOGY")):
            title = line.title(); break
    if not title:
        title = re.sub(r"^\d{4}-\d{2}-\d{2}_", "", filename or "Report").rsplit(".", 1)[0].replace("_", " ")

    # ── Labs ──
    labs: List[Dict[str, Any]] = []
    section_date = None
    in_table = False
    for line in lines:
        low = line.lower()
        if low.startswith("investigation") and "result" in low:
            in_table = True; continue
        if _is_heading(line) and not low.startswith("investigations"):
            in_table = in_table and low.startswith(("blood sugar",))
        hdr_date = re.search(r"\(" + _DATE_RX + r"\)", line)
        if hdr_date and line.split("(")[0].strip().isupper():
            section_date = parse_date(hdr_date.group(0))    # e.g. ADMISSION BLOOD TESTS (22 Aug 2026)
            continue
        if line.isupper() and len(line) < 70 and not re.search(r"\d", line):
            continue                                   # section titles: LIPID PROFILE, URINE ... (not "SGPT (ALT) 22 U/L")
        m = _LAB_RX.match(line)
        if not m or not in_table:
            continue
        name = m.group("name").strip(" .,")
        if _JUNK_NAME.search(name) or len(name) > 45 or len(name) < 2:
            continue
        unit = (m.group("unit") or "").strip()
        rng = m.group("range")
        if not rng and not unit:
            continue                                   # a bare "name number" is too ambiguous
        if unit.lower() in ("tablet", "tablets", "cap", "puffs", "mg") and not rng:
            continue                                   # medicine rows
        low_v = high_v = None
        if rng:
            if "-" in rng:
                a, b = rng.split("-", 1); low_v, high_v = _float(a), _float(b)
            elif rng.strip().startswith(">"):
                low_v = _float(rng)
            else:
                high_v = _float(rng)
        flag = (m.group("flag") or "").lower()
        val = _float(m.group("val"))
        if val is None:
            continue
        status = "high" if flag in ("high", "h") else "low" if flag in ("low", "l") else None
        if status is None and (low_v is not None or high_v is not None):
            status = "low" if (low_v is not None and val < low_v) else "high" if (high_v is not None and val > high_v) else "normal"
        canon = _ALIASES.get(name.lower(), name)
        labs.append({"name": canon, "value": val, "unit": unit, "low": low_v, "high": high_v,
                     "status": status or "normal", "date": section_date or report_date})

    # Spirometry rows: name, predicted, pre-BD, %pred, post-BD, %pred
    for line in lines:
        m = re.match(r"^(FVC \(L\)|FEV1 \(L\)|FEV1/FVC \(%\)|PEF \(L/s\))\s+([\d.]+)\s+([\d.]+)\s+(\d+%|-)\s+([\d.]+)\s+(\d+%|-)", line)
        if m:
            nm = m.group(1).split(" (")[0]
            unit = "%" if "FVC (%)" in m.group(1) else m.group(1).split("(")[1].rstrip(")")
            post, pred = _float(m.group(5)), _float(m.group(2))
            labs.append({"name": f"{nm} (post-BD)", "value": post, "unit": unit, "low": None, "high": None,
                         "status": "low" if (nm == "FEV1/FVC" and post is not None and post < 70) else "normal",
                         "date": report_date, "predicted": pred,
                         "pct_predicted": _float(m.group(6)) if m.group(6) != "-" else None})
    m = re.search(r"Distance walked\s+(\d+)\s*m\s*\((\d+)% of predicted", full)
    if m:
        labs.append({"name": "6-Minute Walk Distance", "value": float(m.group(1)), "unit": "m", "low": None, "high": None,
                     "status": "low" if int(m.group(2)) < 80 else "normal", "date": report_date, "pct_predicted": float(m.group(2))})
    m = re.search(r"CAT score\s+(\d+)\s*/\s*40", full)
    if m:
        labs.append({"name": "CAT score", "value": float(m.group(1)), "unit": "/40", "low": None, "high": 10,
                     "status": "high" if int(m.group(1)) > 10 else "normal", "date": report_date})

    # ── Vitals ──
    vitals: List[Dict[str, Any]] = []
    for i, line in enumerate(lines):
        if not line.lower().startswith("vitals"):
            continue
        seg = line + " " + (lines[i + 1] if i + 1 < len(lines) else "")
        ctx = " ".join(lines[max(0, i - 3):i]).lower()
        when, label = report_date, "Report"
        if "admission" in ctx and admitted:
            when, label = admitted, "On admission"
        elif "discharge" in ctx and discharged:
            when, label = discharged, "At discharge"
        v = {"date": when, "label": label}
        if (m := re.search(r"Temp\.?\s*([\d.]+)\s*°?\s*F", seg, re.I)): v["temperature"] = _float(m.group(1))
        if (m := re.search(r"(?:Pulse|HR|Heart rate)\s*:?\s*(\d+)", seg, re.I)): v["heart_rate"] = int(m.group(1))
        if (m := re.search(r"BP\s*:?\s*(\d{2,3})\s*/\s*(\d{2,3})", seg, re.I)):
            v["systolic_bp"], v["diastolic_bp"] = int(m.group(1)), int(m.group(2))
        if (m := re.search(r"Resp(?:\.|iratory)?\s*rate\s*:?\s*(\d+)", seg, re.I)): v["respiratory_rate"] = int(m.group(1))
        if (m := re.search(r"SpO2?\s*:?\s*(\d{2,3})\s*%", seg, re.I)): v["oxygen_saturation"] = int(m.group(1))
        if len(v) > 2:
            vitals.append(v)
    # PFT / walk test: resting values at the start of the test
    pft = {"date": report_date, "label": "Resting (before walk test)"}
    if (m := re.search(r"SpO2\s+Start\s+(\d{2,3})\s*%", full)): pft["oxygen_saturation"] = int(m.group(1))
    if (m := re.search(r"Heart rate\s+Start\s+(\d+)", full)): pft["heart_rate"] = int(m.group(1))
    if (m := re.search(r"Height\s*/\s*Weight\s+(\d+(?:\.\d+)?)\s*cm\s*/\s*(\d+(?:\.\d+)?)\s*kg", full)):
        pft["height"], pft["weight"] = _float(m.group(1)), _float(m.group(2))
        h = pft["height"] / 100.0
        pft["bmi"] = round(pft["weight"] / (h * h), 1) if h > 0 else None
    if len(pft) > 2:
        vitals.append(pft)
    m = re.search(r"lowest\s+(\d{2,3})\s*%", full)
    exertional_desat = int(m.group(1)) if m and "SpO2" in full else None

    # ── Allergies ──
    allergies = []
    sec = _section(lines, ("known allergies", "allergies"))
    if sec and not re.search(r"\b(nil|none|no known)\b", sec, re.I):
        parts, depth, cur = [], 0, ""
        for ch in sec.rstrip("."):
            if ch == "(": depth += 1
            if ch == ")": depth -= 1
            if ch == "," and depth == 0:
                parts.append(cur); cur = ""
            else:
                cur += ch
        parts.append(cur)
        for p in parts:
            p = p.strip(" .")
            if not p:
                continue
            mm = re.match(r"([^()]+?)\s*(?:\(([^)]*)\))?$", p)
            allergen = (mm.group(1) if mm else p).strip()
            allergies.append({"allergen": allergen.title() if allergen.isupper() else allergen,
                              "reaction": (mm.group(2) or "").strip() if mm else "", "date": report_date})

    # ── Diagnoses ──
    diagnoses = []
    sec = _section(lines, ("final diagnosis", "diagnosis"))
    if sec:
        for item in re.split(r"\s*\b\d+\.\s+", " " + sec):
            item = item.strip(" .")
            if len(item) > 3:
                diagnoses.append(item)
    m = re.search(r"(moderate|mild|severe|very severe) airflow obstruction, GOLD grade\s*(\d)", full, re.I)
    if m and not any("copd" in d.lower() for d in diagnoses):
        diagnoses.append(f"COPD - {m.group(1).lower()} airflow obstruction (GOLD {m.group(2)})")

    notes = {}
    for key, words in (("interpretation", ("interpretation",)), ("impression", ("impression",)),
                       ("recommendations", ("recommendations",)), ("advice", ("advice",)),
                       ("hospital_course", ("hospital course",)), ("discharge_status", ("condition at discharge",)),
                       ("follow_up", ("follow-up",))):
        t = _section(lines, words)
        if t:
            notes[key] = t
    fu = re.search(r"Follow-up:\s*(.+?)(?:_{4,}|$)", full, re.S)
    if fu and "follow_up" not in notes:
        notes["follow_up"] = re.sub(r"\s+", " ", fu.group(1)).strip()[:300]

    return {"title": title, "report_date": report_date, "admitted": admitted, "discharged": discharged,
            "labs": labs, "vitals": vitals, "allergies": allergies, "diagnoses": diagnoses,
            "notes": notes, "exertional_desaturation": exertional_desat}
