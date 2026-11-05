"""Report Ingestion Service v3 — Hardened Pipeline.

Pipeline order:
  1. Extraction Strategy Selection (Text vs Vision)
  2. Map-Reduce Text extraction (for large docs)
  3. Handwriting pass (via Vision)
  4. Hybrid Lab Extraction (Regex + LLM Map-Reduce)
  5. Clinical Context Map-Reduce (Diagnoses/Meds/Notes)
  6. Normalization/Validation
  7. Persistence & Insights
"""
from __future__ import annotations

import os
import json
import re
from datetime import date, datetime
from typing import Optional

from dotenv import load_dotenv
from utils.gemini_client import call_gemini

load_dotenv(override=True)

try:
    import requests
except ImportError:
    requests = None


# ─── New Pipeline ─────────────────────────────────────────────────────────────

def run_hybrid_lab_extraction(
    text: str,
    report_id: int,
    care_recipient_id: int,
    upload_date: Optional[date] = None,
    db=None,
) -> dict:
    """Run the full hybrid lab extraction pipeline on report text.

    Args:
        text:               Raw extracted text from the report.
        report_id:          DB ID of the MedicalReport record.
        care_recipient_id:  Recipient's DB ID.
        upload_date:        Upload date to use as date fallback.
        db:                 SQLAlchemy session (for saving LabValues).

    Returns:
        {
          "report_date":    date | None,
          "template":       str,
          "lab_rows":       list[dict],   # Validated, normalized rows ready for DB
          "unresolved":     list[str],    # Lines that couldn't be parsed
          "sections":       list[str],
          "saved_count":    int,          # How many LabValues were saved
        }
    """
    from services.date_extractor import extract_date_from_text
    from services.document_cleaner import clean_document
    from services.lab_parser import parse_report, parse_report_map_reduce
    from services.lab_normalizer import normalize_and_validate

    # ── Step 1: Deterministic date extraction ─────────────────────────────────
    report_date = extract_date_from_text(text, fallback=upload_date or date.today())
    print(f"[ingestion_v3] Report date: {report_date}")

    # ── Step 2: Parse (clean + section + extract + LLM fallback Map-Reduce) ───
    words = text.split()
    if len(words) > 4000:
        parsed = parse_report_map_reduce(text)
    else:
        parsed = parse_report(text)
        
    template   = parsed["template"]
    raw_rows   = parsed["rows"]
    unresolved = parsed.get("unresolved", [])
    sections   = parsed["section_labels"]
    cleaned    = clean_document(text)

    print(f"[ingestion_v3] Template={template}, raw_rows={len(raw_rows)}")

    # ── Step 3: Normalize + validate each row ─────────────────────────────────
    validated_rows: list[dict] = []
    rejected_rows: list[dict] = []
    
    for row in raw_rows:
        norm = normalize_and_validate(
            raw_name=row.get("raw_name", ""),
            raw_value=row.get("value", 0.0),
            raw_unit=row.get("unit", ""),
            source=row.get("source", "regex"),
            db=db
        )
        if norm is None:
            rejected_rows.append(row)
            continue   
            
        norm.update({
            "source_text":   row.get("source_text", ""),
            "ref_low":       row.get("ref_low"),
            "ref_high":      row.get("ref_high"),
            "section":       row.get("section", "GENERAL"),
            "report_date":   report_date,
            "raw_metric_name": row.get("raw_name"),
        })
        validated_rows.append(norm)

    print(f"[ingestion_v2] Funnel: {len(raw_rows)} raw -> {len(validated_rows)} validated -> {len(rejected_rows)} rejected")
    if rejected_rows:
        print(f"[ingestion_v2] REJECTED ROWS: {json.dumps(rejected_rows, indent=2)}")

    # ── Step 4: Save LabValues to DB ──────────────────────────────────────────
    saved_count = 0
    if db is not None and validated_rows:
        saved_count = _save_lab_values(
            db, validated_rows, report_id, care_recipient_id, report_date
        )

    # ── Step 5: RAG indexing (best-effort, non-blocking) ─────────────────────
    try:
        from services.rag_service import index_report_chunks
        index_report_chunks(
            report_id=report_id,
            care_recipient_id=care_recipient_id,
            cleaned_lines=cleaned["cleaned_lines"],
            extracted_rows=raw_rows,
        )
    except Exception as e:
        print(f"[ingestion_v2] RAG indexing failed (non-fatal): {e}")

    return {
        "report_date":  report_date,
        "template":     template,
        "lab_rows":     validated_rows,
        "unresolved":   unresolved,
        "sections":     sections,
        "saved_count":  saved_count,
    }


def _save_lab_values(
    db,
    validated_rows: list[dict],
    report_id: int,
    care_recipient_id: int,
    report_date: date,
) -> int:
    """Persist validated lab rows to the LabValue table.
    
    Deduplicates: if a row with the same (metric_name, recorded_date, report_id)
    already exists, it is skipped.
    
    Returns number of rows saved.
    """
    from tables.medical_conditions import LabValue

    saved = 0
    for row in validated_rows:
        metric = row["metric_name"]
        rec_date = row.get("report_date") or report_date

        # Duplicate check
        existing = db.query(LabValue).filter(
            LabValue.care_recipient_id == care_recipient_id,
            LabValue.metric_name == metric,
            LabValue.recorded_date == rec_date,
            LabValue.report_id == report_id,
        ).first()
        if existing:
            continue

        # Abnormal detection
        ref_low  = row.get("ref_low")
        ref_high = row.get("ref_high")
        norm_val = row["normalized_value"]
        is_abnormal = False
        if ref_low is not None and norm_val < ref_low:
            is_abnormal = True
        if ref_high is not None and norm_val > ref_high:
            is_abnormal = True

        lv = LabValue(
            care_recipient_id   = care_recipient_id,
            report_id           = report_id,
            metric_name         = metric,
            raw_metric_name     = row.get("raw_metric_name"),
            metric_value        = row["metric_value"],
            unit                = row["unit"],
            normalized_value    = norm_val,
            normalized_unit     = row["normalized_unit"],
            reference_range_low = ref_low,
            reference_range_high= ref_high,
            is_abnormal         = is_abnormal,
            is_mapped           = row.get("is_mapped", True),
            needs_review        = row.get("needs_review", False),
            recorded_date       = rec_date,
            source_text         = row.get("source_text", "")[:500] if row.get("source_text") else None,
            confidence_score    = row.get("confidence_score", 0.9),
            extraction_source   = row.get("extraction_source", "regex"),
        )
        db.add(lv)
        saved += 1

    if saved:
        db.flush()

    return saved


def run_vision_lab_extraction(
    image_bytes: bytes,
    mime_type: str,
    report_id: int,
    care_recipient_id: int,
    upload_date: Optional[date] = None,
    db=None,
) -> dict:
    """Run extraction via Gemini Vision (Multimodal) for scanned/image reports."""
    from utils.summarizer import vision_extract_report
    from services.lab_normalizer import normalize_and_validate
    
    print(f"[ingestion_v3] Using Vision pipeline for report {report_id}...")
    extracted = vision_extract_report(image_bytes, mime_type)
    
    raw_rows = extracted.get("lab_values", [])
    report_date = upload_date or date.today()
    
    validated_rows = []
    for row in raw_rows:
        norm = normalize_and_validate(
            raw_name=row.get("raw_name", ""),
            raw_value=row.get("value", 0.0),
            raw_unit=row.get("unit", ""),
            source="vision",
            db=db
        )
        if norm:
            norm.update({
                "source_text":   f"Vision extraction: {row.get('flag', 'normal')}",
                "ref_low":       row.get("ref_low"),
                "ref_high":      row.get("ref_high"),
                "section":       "VISION",
                "report_date":   report_date,
                "raw_metric_name": row.get("raw_name"),
            })
            validated_rows.append(norm)

    saved_count = 0
    if db is not None and validated_rows:
        saved_count = _save_lab_values(db, validated_rows, report_id, care_recipient_id, report_date)

    return {
        "report_date":      report_date,
        "template":         "VISION_MULTIMODAL",
        "lab_rows":         validated_rows,
        "clinical_notes":   extracted.get("clinical_notes", ""),
        "sections":         ["VISION"],
        "saved_count":      saved_count
    }


# ─── Clinical Context Extraction (Diagnoses, Medications, Notes) ──────────────
# Handles map-reduce to ensure no clinical context is lost in large docs.

def extract_structured_report(text: str, report_date_hint: str = None) -> dict:
    """Legacy wrapper / entry point for clinical context extraction."""
    words = (text or "").split()
    if len(words) > 5000:
        return extract_structured_report_map_reduce(text, report_date_hint)
    return _extract_structured_report_single_chunk(text, report_date_hint)


def extract_structured_report_map_reduce(text: str, report_date_hint: str = None) -> dict:
    """Chunked clinical context extraction."""
    from utils.summarizer import chunk_text
    chunks = chunk_text(text, chunk_size=3000, overlap=300)
    
    all_diagnoses = set()
    all_resolved = set()
    all_medications = set()
    all_symptoms = set()
    all_notes = []
    all_med_details = {}

    print(f"[ingestion_v3] Clinical context map-reduce ({len(chunks)} chunks)")
    for i, chunk in enumerate(chunks):
        res = _extract_structured_report_single_chunk(chunk, report_date_hint)
        all_diagnoses.update(res.get("diagnoses", []))
        all_resolved.update(res.get("resolved_diagnoses", []))
        all_medications.update(m for m in res.get("medications", []) if isinstance(m, str))
        for d in res.get("medication_details", []) or []:
            if isinstance(d, dict) and d.get("name"):
                all_med_details.setdefault(str(d["name"]).strip().lower(), d)
        all_symptoms.update(res.get("symptoms", []))
        if res.get("doctor_notes"):
            all_notes.append(res["doctor_notes"])
            
    return {
        "report_date":        report_date_hint or str(date.today()),
        "diagnoses":          list(all_diagnoses),
        "resolved_diagnoses": list(all_resolved),
        "lab_values":         {},
        "medications":        list(all_medications),
        "medication_details": list(all_med_details.values()),
        "doctor_notes":       " | ".join(all_notes),
        "symptoms":           list(all_symptoms)
    }


def _extract_structured_report_single_chunk(text: str, report_date_hint: str = None) -> dict:
    """Extract diagnoses, medications, doctor_notes, and symptoms via Gemini.

    Lab values are no longer extracted here — they come from run_hybrid_lab_extraction().
    This function now fills the 'clinical context' part of the extracted_data JSON.
    """
    empty_result = {
        "report_date": report_date_hint or str(date.today()),
        "diagnoses": [],
        "resolved_diagnoses": [],
        "lab_values": {},       # Left empty — hybrid pipeline handles this
        "medications": [],
        "doctor_notes": "",
        "symptoms": [],
    }

    if not text or not text.strip():
        return empty_result

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key or not requests:
        return empty_result

    # Truncate text to essentials (reduce token cost)
    # For diagnosis detection we mostly need the first and last sections
    truncated = text[:4000] + ("\n...\n" + text[-2000:] if len(text) > 6000 else "")

    prompt = """You are a clinical data extractor. From the medical report below, extract ONLY:
- Active diagnoses / diseases explicitly mentioned
- Resolved / past diagnoses mentioned
- Medications listed (name + dose if available)
- Key doctor observations or impressions (brief)
- Symptoms reported by the patient

Return a VALID JSON object with EXACTLY this structure (no markdown, no code fences):
{
  "report_date": "YYYY-MM-DD or empty string",
  "diagnoses": ["Disease Name 1"],
  "resolved_diagnoses": ["Past Disease"],
  "lab_values": {},
  "medications": ["Medication 1 with dose"],
  "medication_details": [{"name": "Metformin", "dosage": "500 mg", "frequency": "twice daily", "timing": "after breakfast and dinner", "duration": "ongoing", "status": "active"}],
  "doctor_notes": "Brief clinical impressions",
  "symptoms": ["Symptom 1"]
}

RULES:
0. medication_details: one entry per medicine the patient should take at home (prescription,
   discharge medications, current medication list). Use "status": "stopped" for medicines marked
   stopped/discontinued. Do NOT include drugs only given in hospital (IV/injections during admission).
   Use "" for unknown fields.
1. Leave lab_values as empty {} — they are handled separately
2. Only include diagnoses explicitly mentioned or clearly implied
3. Return null fields as empty lists/strings, NOT as null
4. Return ONLY valid JSON, no other text

Medical Report:
""" + truncated

    try:
        print(f"[ingestion_v2] Gemini clinical context extraction ({len(text)} chars)")
        data = call_gemini(
            {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"maxOutputTokens": 2000, "temperature": 0.05, "topP": 0.8},
            },
            timeout=60,
            caller="[ingestion_v2/gemini_context]",
        )

        if data and "candidates" in data and data["candidates"]:
            raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
            raw_text = raw_text.strip()
            cleaned = re.sub(r"```(?:json)?\s*", "", raw_text, flags=re.DOTALL)
            cleaned = re.sub(r"\s*```", "", cleaned, flags=re.DOTALL).strip()

            try:
                result = json.loads(cleaned)
                result.setdefault("report_date", report_date_hint or str(date.today()))
                result.setdefault("diagnoses", [])
                result.setdefault("resolved_diagnoses", [])
                result["lab_values"] = {}   # Always override — we don't want Gemini lab guesses
                result.setdefault("medications", [])
                result.setdefault("doctor_notes", "")
                result.setdefault("symptoms", [])
                print(f"[ingestion_v2] Gemini context: {len(result['diagnoses'])} diagnoses")
                return result
            except json.JSONDecodeError:
                # Try extracting first { ... } block
                m = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
                if m:
                    try:
                        result = json.loads(m.group(0))
                        result["lab_values"] = {}
                        return result
                    except json.JSONDecodeError:
                        pass

        return empty_result

    except Exception as e:
        print(f"[ingestion_v2] Gemini context extraction failed: {e}")
        return empty_result
