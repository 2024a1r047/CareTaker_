"""
Report -> Allergies sync.

Pulls allergies out of an uploaded medical report's text and adds them to the
care recipient's Allergies list (the same `allergies` table used by the profile
page's manual "Add Allergy" form).

Handles the formats seen in real reports:
  A) Section style       "Known Allergies\n PENICILLIN (urticaria), NSAIDs (bronchospasm)."
  B) Table style         "Allergen  Type  Reaction  Severity\n Penicillin group  Drug  Skin rash  Mild"
  C) Free-text mentions  "allergic to sulfa drugs", "allergy to peanuts", "in view of penicillin allergy"

Negations such as "None known", "NKDA", "No known food allergies" are ignored.
Existing allergies are never modified or deleted — only new ones are added.
"""
import re
import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

# ── Keyword tables ──────────────────────────────────────────────────────────
_DRUG_WORDS = (
    "penicillin", "cillin", "amoxi", "ampi", "cephalo", "cef", "sulfa", "sulfon", "sulpha", "nsaid",
    "aspirin", "ibuprofen", "diclofenac", "naproxen", "codeine", "morphine", "opioid", "tramadol",
    "mycin", "cycline", "floxacin", "statin", "iodine", "contrast", "antibiotic", "drug", "vaccine",
    "insulin", "metformin", "heparin", "warfarin", "ace inhibitor", "pril", "paracetamol", "acetaminophen",
)
_FOOD_WORDS = (
    "peanut", "nut", "almond", "cashew", "walnut", "egg", "milk", "dairy", "lactose", "gluten", "wheat",
    "shellfish", "shrimp", "prawn", "crab", "lobster", "fish", "soy", "mushroom", "sesame", "strawberr",
    "tomato", "banana", "chocolate", "seafood", "food", "brinjal", "eggplant", "mustard", "corn",
)
_ENV_WORDS = (
    "dust", "mite", "pollen", "mold", "mould", "pet", "cat", "dog", "dander", "latex", "bee", "wasp",
    "insect", "grass", "smoke", "perfume", "fragrance", "cockroach", "feather", "nickel",
)
_KNOWN_ALLERGENS = _DRUG_WORDS + _FOOD_WORDS + _ENV_WORDS

_NEGATION = re.compile(
    r"\b(none\s+known|no\s+known|nkda|nka|nil\b|none\b|not\s+known|denies|no\s+allerg|contains\s+no|negative)\b",
    re.I,
)
_BAD_TOKENS = re.compile(
    r"\b(diagnosis|history|contact|emergency|patient|page|report|clinic|hospital|medication|dose|daily|"
    r"tablet|mg|mcg|ml|advice|follow|review|blood|pressure|sugar|parameter|reading|remark|age|sex|"
    r"date|doctor|signature|smoker|smoking|avoid|reason|type|reaction|severity|allergen|allergies)\b",
    re.I,
)
_SEVERITY_WORDS = r"(?:mild\s+to\s+moderate|moderate\s+to\s+severe|life[-\s]threatening|mild|moderate|severe|critical|high|low)"
_TYPE_WORDS = r"(?:drug|medication|medicine|food|environmental|environment|other|contact|insect)"


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip(" .,:;-–")


def _valid_allergen(name: str) -> bool:
    n = _clean(name)
    if not (2 <= len(n) <= 45):
        return False
    if len(n.split()) > 6 or re.search(r"\d", n) or ":" in n:
        return False
    if _NEGATION.search(n) or _BAD_TOKENS.search(n):
        return False
    return bool(re.search(r"[A-Za-z]{3,}", n))


def _pretty(name: str) -> str:
    n = _clean(name)
    if n.isupper() and len(n) > 5:          # "PENICILLIN" -> "Penicillin" (keep "NSAIDs"-style acronyms)
        n = n.title()
    return n[0].upper() + n[1:] if n else n


def infer_type(name: str, explicit: Optional[str] = None) -> str:
    if explicit:
        e = explicit.lower()
        if e in ("drug", "medication", "medicine"):
            return "drug"
        if e == "food":
            return "food"
        if e in ("environmental", "environment", "insect", "contact"):
            return "environment"
    low = (name or "").lower()
    if any(w in low for w in _DRUG_WORDS):
        return "drug"
    if any(w in low for w in _FOOD_WORDS):
        return "food"
    if any(w in low for w in _ENV_WORDS):
        return "environment"
    return "other"


def infer_severity(reaction: str, explicit: Optional[str] = None) -> str:
    if explicit:
        e = explicit.lower()
        if "life" in e or "critical" in e:
            return "critical"
        if "severe" in e or e == "high":
            return "severe"
        if "moderate" in e:
            return "moderate"
        if "mild" in e or e == "low":
            return "mild"
    r = (reaction or "").lower()
    if "anaphyla" in r:
        return "critical"
    if any(w in r for w in ("bronchospasm", "angioedema", "swelling", "breath", "wheez", "shock")):
        return "severe"
    if any(w in r for w in ("urticaria", "hives", "vomit")):
        return "moderate"
    if any(w in r for w in ("rash", "itch", "sneez", "redness")):
        return "mild"
    return "moderate"


# ── Extraction ──────────────────────────────────────────────────────────────
def _split_top_level(text: str) -> List[str]:
    parts, depth, cur = [], 0, ""
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        if ch in ",;" and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    return parts


def extract_allergies_from_text(text: str) -> List[Dict[str, Any]]:
    """Return a de-duplicated list of {allergen, allergy_type, reaction, severity}."""
    if not text:
        return []
    lines = [l.strip() for l in text.splitlines()]
    found: List[Dict[str, Any]] = []

    def add(name, reaction="", a_type=None, severity=None):
        if not _valid_allergen(name):
            return
        name = _pretty(name)
        reaction = _clean(reaction)[:120]
        found.append({
            "allergen": name,
            "allergy_type": infer_type(name, a_type),
            "reaction": reaction or None,
            "severity": infer_severity(reaction, severity),
        })

    # B) Table rows under an "Allergen ... Type ... Reaction ... Severity" header
    row_re = re.compile(rf"^(.+?)\s+({_TYPE_WORDS})\s+(.+?)\s+({_SEVERITY_WORDS})\s*$", re.I)
    for i, line in enumerate(lines):
        if re.search(r"allergen", line, re.I) and re.search(r"reaction", line, re.I):
            for row in lines[i + 1:i + 15]:
                if not row:
                    continue
                m = row_re.match(row)
                if not m:
                    if _NEGATION.search(row) or len(row) > 160:
                        break
                    continue
                add(m.group(1), m.group(3), m.group(2), m.group(4))

    # A) "Known Allergies" / "Allergies" section followed by a comma list
    for i, line in enumerate(lines):
        if not re.fullmatch(r"(known\s+)?(drug\s+)?allerg(y|ies)\s*:?", line, re.I):
            continue
        body = ""
        for nxt in lines[i + 1:i + 4]:
            if nxt:
                body = nxt
                break
        if not body or _NEGATION.search(body) or re.search(r"allergen", body, re.I):
            continue
        for part in _split_top_level(body.rstrip(".")):
            m = re.match(r"\s*([^()]+?)\s*(?:\(([^)]*)\))?\s*$", part)
            if m:
                add(m.group(1), m.group(2) or "")

    # Inline label: "Allergies: Penicillin, Sulfa" (same line)
    for m in re.finditer(r"(?:known\s+)?allerg(?:y|ies)\s*[:\-]\s*([^\n]{3,160})", text, re.I):
        body = m.group(1)
        if _NEGATION.search(body):
            continue
        for part in _split_top_level(body.split(".")[0]):
            mm = re.match(r"\s*([^()]+?)\s*(?:\(([^)]*)\))?\s*$", part)
            if mm:
                add(mm.group(1), mm.group(2) or "")

    # C) Free text: "allergic to X" / "allergy to X" / "<known allergen> allergy"
    for sentence in re.split(r"(?<=[.!?])\s+|\n", text):
        if _NEGATION.search(sentence):
            continue
        for m in re.finditer(r"allerg(?:ic|y)\s+to\s+([^.;()\n]{2,100})", sentence, re.I):
            body = re.split(r"\b(?:with|causing|which|leading|resulting|since|for|in)\b", m.group(1), flags=re.I)[0]
            for part in re.split(r",|\band\b|\bor\b", body, flags=re.I):
                add(part)
        for m in re.finditer(r"\b([A-Za-z][A-Za-z\-]{2,25})\s+allerg(?:y|ies)\b", sentence, re.I):
            word = m.group(1)
            if any(k in word.lower() for k in _KNOWN_ALLERGENS) and word.lower() not in ("food", "drug"):
                add(word)

    # De-duplicate: prefer the entry with the most detail (reaction present)
    unique: Dict[str, Dict[str, Any]] = {}
    for a in found:
        key = _norm_key(a["allergen"])
        dup = next((k for k in unique if _same_allergen(k, key)), None)
        if dup is None:
            unique[key] = a
        elif not unique[dup].get("reaction") and a.get("reaction"):
            unique[dup] = a
    return list(unique.values())


def _norm_key(name: str) -> str:
    return re.sub(r"[^a-z ]", "", (name or "").lower()).strip()


def _same_allergen(a: str, b: str) -> bool:
    """'penicillin' == 'penicillin group of antibiotics'; 'nsaids' == 'nsaid'."""
    a, b = a.rstrip("s"), b.rstrip("s")
    if a == b:
        return True
    fa, fb = (a.split() or [""])[0], (b.split() or [""])[0]
    return len(fa) >= 4 and fa == fb


# ── Persistence ─────────────────────────────────────────────────────────────
def sync_report_allergies(report, db: Session, text: Optional[str] = None) -> Dict[str, Any]:
    """Extract allergies from a report and add any that the recipient doesn't already have."""
    from tables.allergies import Allergy, AllergyType, AllergyStatus

    if text is None:
        text = ""
        if getattr(report, "data", None):
            try:
                from utils.summarizer import extract_text_from_bytes
                text = extract_text_from_bytes(report.data, report.filename) or ""
            except Exception as e:
                print(f"[allergy_sync] text extraction failed for report {report.id}: {e}")

    candidates = extract_allergies_from_text(text)
    if not candidates:
        return {"report_id": report.id, "found": 0, "added": []}

    existing = db.query(Allergy).filter(Allergy.care_recipient_id == report.care_recipient_id).all()
    existing_keys = [_norm_key(a.allergen) for a in existing]

    report_date = getattr(report, "report_date", None)
    if isinstance(report_date, datetime.datetime):
        report_date = report_date.date()

    added = []
    for c in candidates:
        key = _norm_key(c["allergen"])
        if any(_same_allergen(key, k) for k in existing_keys):
            continue
        db.add(Allergy(
            care_recipient_id=report.care_recipient_id,
            allergen=c["allergen"],
            allergy_type=AllergyType(c["allergy_type"]),
            reaction=c["reaction"],
            severity=c["severity"],
            diagnosed_date=report_date if isinstance(report_date, datetime.date) else None,
            status=AllergyStatus.active,
        ))
        existing_keys.append(key)
        added.append(c["allergen"])

    if added:
        db.commit()
        print(f"[allergy_sync] report {report.id}: added {added}")
    return {"report_id": report.id, "found": len(candidates), "added": added}


def sync_recipient_allergies_from_reports(recipient_id: int, db: Session) -> Dict[str, Any]:
    """Run the allergy sync over every stored report of a care recipient."""
    from tables.medical_reports import MedicalReport

    reports = db.query(MedicalReport).filter(MedicalReport.care_recipient_id == recipient_id).all()
    added: List[str] = []
    for r in reports:
        try:
            res = sync_report_allergies(r, db)
            added.extend(res["added"])
        except Exception as e:
            db.rollback()
            print(f"[allergy_sync] report {r.id} failed: {e}")
    return {"reports_scanned": len(reports), "added": added}
