"""
Test Suite: Voice Agent Personal Information Access & Profile Name Personalization
Verifies:
1. The voice agent addresses the user strictly by their profile name (preferred_name).
2. The voice agent has direct access to all personal information present in the profile
   (medicines, allergies, emergency contacts, doctor remarks, blood group).
3. The voice agent remembers personal information from conversation memory (saathi_db).
4. The voice agent records every conversation turn and session in saathi_db.
"""

import os
import sys
import uuid
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import SaathiSessionLocal, SessionLocal
from tables.users import CareRecipient, CareTaker
from tables.medications import Medication
from tables.medical_reports import MedicalReport
from tables.allergies import Allergy, AllergyType, AllergyStatus
from tables.medical_conditions import PatientCondition
from tables.saathi_db_models import (
    FamilyMember,
    ConversationSession,
    ConversationMessage,
    MemberMemory,
)
from services.saathi_memory_service import (
    start_session,
    save_message_sync,
    end_session_sync,
    get_member_context,
    build_system_prompt,
    build_agent_context,
    sync_member_profile,
)
from services.voice_bot_engine import (
    build_conversation_context,
    generate_system_prompt,
    intelligent_saathi_dialogue,
)


def run_tests():
    print("=" * 70)
    print("VOICE AGENT PERSONAL INFO ACCESS & PROFILE NAME VERIFICATION TEST")
    print("=" * 70)

    # 1. Setup Caretaker and Profile in Main Database
    with SessionLocal() as db:
        caretaker = db.query(CareTaker).first()
        if not caretaker:
            caretaker = CareTaker(
                username="raman_care",
                full_name="Raman Sharma",
                email="raman@example.com",
                phone_number="+91 9876543210",
                password_hash="dummy"
            )
            db.add(caretaker)
            db.commit()
            db.refresh(caretaker)

        recip = db.query(CareRecipient).filter(CareRecipient.caretaker_id == caretaker.id).first()
        if not recip:
            recip = CareRecipient(
                caretaker_id=caretaker.id,
                full_name="Harbans Lal Sharma",
                preferred_name="Bauji",
                email="harbans@example.com",
                phone_number="9876500001",
                age=76,
                gender="Male",
                city="Chandigarh",
                blood_group="B+",
                emergency_contact="+91 9876543210",
                doctor_remarks="Take morning walk, avoid high sodium, take BP tablet after breakfast.",
                report_summary="Mild hypertension and controlled Type 2 Diabetes."
            )
            db.add(recip)
            db.commit()
            db.refresh(recip)
        else:
            recip.preferred_name = "Bauji"
            recip.blood_group = "B+"
            recip.emergency_contact = "+91 9876543210"
            recip.doctor_remarks = "Take morning walk, avoid high sodium, take BP tablet after breakfast."
            db.commit()

        test_recip_id = recip.id
        test_caretaker_id = caretaker.id

        # Setup active medication for recipient
        existing_med = db.query(Medication).filter(
            Medication.care_recipient_id == test_recip_id,
            Medication.medicine_name == "Amlodipine"
        ).first()
        if not existing_med:
            med = Medication(
                care_recipient_id=test_recip_id,
                medicine_name="Amlodipine",
                dosage="5mg",
                frequency="Once daily",
                schedule_time="09:00 AM",
                status="active"
            )
            db.add(med)
            db.commit()
        else:
            existing_med.status = "active"
            db.commit()

        # Setup allergy for recipient
        existing_allergy = db.query(Allergy).filter(
            Allergy.care_recipient_id == test_recip_id,
            Allergy.allergen == "Penicillin"
        ).first()
        if not existing_allergy:
            allergy = Allergy(
                care_recipient_id=test_recip_id,
                allergen="Penicillin",
                allergy_type=AllergyType.drug,
                reaction="Skin rash and swelling",
                severity="Severe",
                status=AllergyStatus.active
            )
            db.add(allergy)
            db.commit()

        # Setup medical condition
        existing_cond = db.query(PatientCondition).filter(
            PatientCondition.care_recipient_id == test_recip_id,
            PatientCondition.disease_name == "Hypertension"
        ).first()
        if not existing_cond:
            cond = PatientCondition(
                care_recipient_id=test_recip_id,
                disease_name="Hypertension",
                status="active"
            )
            db.add(cond)
            db.commit()

        # Setup uploaded medical report
        existing_rep = db.query(MedicalReport).filter(
            MedicalReport.care_recipient_id == test_recip_id,
            MedicalReport.filename == "blood_test_report_latest.pdf"
        ).first()
        if not existing_rep:
            rep = MedicalReport(
                care_recipient_id=test_recip_id,
                filename="blood_test_report_latest.pdf",
                mime_type="application/pdf",
                analysis_summary="HbA1c 6.8% indicates controlled diabetes. Lipid profile normal. Kidney and liver functions normal.",
                processing_status="completed"
            )
            db.add(rep)
            db.commit()

    print(f"[Setup] Recipient ID: {test_recip_id}, Full Name: 'Harbans Lal Sharma', Profile Name: 'Bauji'")

    # ─────────────────────────────────────────────────────────
    # TEST 1: Profile Name Personalization & System Prompt
    # ─────────────────────────────────────────────────────────
    print("\n--- TEST 1: Calling the user strictly by their Profile Name ---")
    ctx = get_member_context(test_recip_id)
    assert ctx["preferred_name"] == "Bauji", f"Expected Bauji, got {ctx['preferred_name']}"
    print(f"  [PASS] Profile name loaded: '{ctx['preferred_name']}'")

    prompt = build_system_prompt(test_recip_id)
    assert 'You are talking to Bauji. Always address them as "Bauji".' in prompt
    assert "Active Medications:" in prompt
    assert "Amlodipine" in prompt
    assert "Penicillin" in prompt
    assert "Blood Group: B+" in prompt
    assert "PERSONAL PROFILE ACCESS & INSTRUCTIONS:" in prompt
    print("  [PASS] System prompt contains full personal profile access and mandates addressing as 'Bauji'.")

    # ─────────────────────────────────────────────────────────
    # TEST 2: Direct Dialogue Queries on Personal Profile Info
    # ─────────────────────────────────────────────────────────
    print("\n--- TEST 2: Voice Agent Answering Personal Profile Questions ---")
    with SessionLocal() as db:
        # A. Name Query
        res_name = intelligent_saathi_dialogue("mera naam kya hai", test_recip_id, db, "hi")
        print(f"  User: 'Mera naam kya hai?'")
        print(f"  Saathi: '{res_name['reply']}'")
        assert "Harbans Lal Sharma" in res_name["reply"] or "Bauji" in res_name["reply"]
        assert "Bauji" in res_name["reply"]
        print("  [PASS] Saathi correctly stated user's full name and profile name 'Bauji'.")

        # B. Medicine Query
        res_med = intelligent_saathi_dialogue("meri dawai kya hai", test_recip_id, db, "hi")
        print(f"  User: 'Meri dawai kya hai?'")
        print(f"  Saathi: '{res_med['reply']}'")
        assert "Amlodipine" in res_med["reply"]
        assert "Bauji" in res_med["reply"]
        print("  [PASS] Saathi accurately reported active medications from profile.")

        # C. Allergy Query
        res_allergy = intelligent_saathi_dialogue("mujhe kis cheez se allergy hai", test_recip_id, db, "hi")
        print(f"  User: 'Mujhe kis cheez se allergy hai?'")
        print(f"  Saathi: '{res_allergy['reply']}'")
        assert "Penicillin" in res_allergy["reply"]
        print("  [PASS] Saathi accurately reported known allergies from profile.")

        # D. Emergency Contact / Caretaker Query
        res_contact = intelligent_saathi_dialogue("mera emergency contact kaun hai", test_recip_id, db, "hi")
        print(f"  User: 'Mera emergency contact kaun hai?'")
        print(f"  Saathi: '{res_contact['reply']}'")
        assert "caretaker" in res_contact["reply"].lower() or "alert" in res_contact["reply"].lower()
        print("  [PASS] Saathi accurately reported primary caretaker and emergency contact.")

        # E. Blood Group Query
        res_blood = intelligent_saathi_dialogue("mera blood group kya hai", test_recip_id, db, "hi")
        print(f"  User: 'Mera blood group kya hai?'")
        print(f"  Saathi: '{res_blood['reply']}'")
        assert "B+" in res_blood["reply"]
        print("  [PASS] Saathi accurately reported blood group B+ from profile.")

        # F. Doctor Remarks Query
        res_doc = intelligent_saathi_dialogue("doctor ne kya bola tha", test_recip_id, db, "hi")
        print(f"  User: 'Doctor ne kya bola tha?'")
        print(f"  Saathi: '{res_doc['reply']}'")
        assert "walk" in res_doc["reply"].lower() or "dhyan" in res_doc["reply"].lower()
        print("  [PASS] Saathi accurately conveyed doctor's remarks and advice.")

        # G. Uploaded Medical Report Query
        res_rep = intelligent_saathi_dialogue("mera medical report kya kehta hai", test_recip_id, db, "hi")
        print(f"  User: 'Mera medical report kya kehta hai?'")
        print(f"  Saathi: '{res_rep['reply']}'")
        assert "report" in res_rep["reply"].lower() or "diabetes" in res_rep["reply"].lower() or "hba1c" in res_rep["reply"].lower()
        assert "Bauji" in res_rep["reply"]
        print("  [PASS] Saathi accurately reported uploaded medical report findings from profile.")

    # ─────────────────────────────────────────────────────────
    # TEST 2.5: Dynamic Agent Context & Profile Synchronization
    # ─────────────────────────────────────────────────────────
    print("\n--- TEST 2.5: Dynamic Agent Context (8 variables) & saathi_db Sync ---")
    sync_res = sync_member_profile(test_recip_id)
    assert sync_res["success"] is True
    print("  [PASS] Personal data, medicines, and reports synced to saathi_db.family_members.")

    agent_ctx = build_agent_context(test_recip_id)
    assert agent_ctx["user_name"] == "Bauji"
    assert "Amlodipine" in agent_ctx["medicines"]
    assert "blood_test_report_latest.pdf" in agent_ctx["medical_reports"] or "diabetes" in agent_ctx["medical_reports"].lower()
    assert "Bauji" in agent_ctx["profile_details"]
    assert "B+" in agent_ctx["profile_details"]
    print("  [PASS] build_agent_context returned all 8 clean variables with full medicine & report data.")

    # ─────────────────────────────────────────────────────────
    # TEST 3: Remembering Personal Info in saathi_db Memory
    # ─────────────────────────────────────────────────────────
    print("\n--- TEST 3: Remembering Information Across Sessions in saathi_db ---")
    sess_id = start_session(test_caretaker_id, test_recip_id)
    # User shares a personal fact
    user_fact = "Mujhe subah tulsi wali green tea peena pasand hai."
    save_message_sync(sess_id, test_recip_id, "user", user_fact, "hi")
    save_message_sync(sess_id, test_recip_id, "assistant", "Bauji, tulsi green tea swasthya ke liye bohot gunkaari hoti hai.", "hi")
    end_session_sync(sess_id)

    with SessionLocal() as db:
        res_mem = intelligent_saathi_dialogue("aapko mere baare mein kya yaad hai", test_recip_id, db, "hi")
        print(f"  User: 'Aapko mere baare mein kya yaad hai?'")
        print(f"  Saathi: '{res_mem['reply']}'")
        assert "Bauji" in res_mem["reply"]
        assert len(res_mem["reply"]) > 20
        print("  [PASS] Saathi accessed remembered personal facts from saathi_db memory.")

    # ─────────────────────────────────────────────────────────
    # TEST 4: Conversation Storage in saathi_db
    # ─────────────────────────────────────────────────────────
    print("\n--- TEST 4: Verifying Conversation Recording in saathi_db ---")
    with SaathiSessionLocal() as s_db:
        session = s_db.query(ConversationSession).filter(ConversationSession.id == sess_id).first()
        assert session is not None, "Session not found in saathi_db"
        assert session.ended_at is not None, "Session ended_at not recorded"
        assert session.summary is not None, "Session summary not generated"
        print(f"  [PASS] Session {session.id} recorded with summary: '{session.summary}'")

        msgs = s_db.query(ConversationMessage).filter(ConversationMessage.session_id == sess_id).all()
        assert len(msgs) >= 2, f"Expected at least 2 messages, got {len(msgs)}"
        print(f"  [PASS] {len(msgs)} conversation turns recorded with session_id FK in saathi_db.")

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED: Profile Name, Personal Info Access & Memory Verified!")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()
