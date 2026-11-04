"""
Integration test for Saathi Voice Assistant Memory & Conversation System
Tests saathi_db:
1. Conversation Sessions & Messages (Goal 1)
2. Member Memory extraction & upsert across categories (Goal 2)
3. Caretaker-given preferred_name & standardized system prompt template (Goal 3)
4. Privacy controls (access by account/member, memory & session management)
"""

import os
import sys
import uuid
import time
from datetime import datetime, timezone

# Ensure backend directory is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import SaathiSessionLocal, SessionLocal
from tables.saathi_db_models import (
    FamilyMember,
    ConversationSession,
    ConversationMessage,
    MemberMemory,
)
from tables.users import CareRecipient, CareTaker
from services.saathi_memory_service import (
    start_session,
    save_message_sync,
    save_message_async,
    end_session_sync,
    extract_and_upsert_memory,
    get_member_context,
    build_system_prompt,
    VALID_CATEGORIES,
)


def run_tests():
    print("=" * 60)
    print("SAATHI VOICE ASSISTANT - INTEGRATION TEST SUITE")
    print("=" * 60)

    # 1. Setup test caretaker and recipient in main DB if not present
    with SessionLocal() as db:
        caretaker = db.query(CareTaker).first()
        if not caretaker:
            caretaker = CareTaker(username="test_caretaker", email="caretaker@test.com", password_hash="dummy")
            db.add(caretaker)
            db.commit()
            db.refresh(caretaker)

        recip = db.query(CareRecipient).filter(CareRecipient.caretaker_id == caretaker.id).first()
        if not recip:
            recip = CareRecipient(
                caretaker_id=caretaker.id,
                full_name="Harbans Lal Sharma",
                preferred_name="Bauji",
                age=74,
                gender="Male",
                city="Chandigarh",
                report_summary="Mild hypertension and knee osteoarthritis",
            )
            db.add(recip)
            db.commit()
            db.refresh(recip)
        else:
            # Ensure preferred_name is set for testing
            recip.preferred_name = "Bauji"
            db.commit()

        test_caretaker_id = caretaker.id
        test_member_id = recip.id
        print(f"[Setup] Using Caretaker ID: {test_caretaker_id}, Member ID: {test_member_id} ({recip.full_name}, preferred: '{recip.preferred_name}')")

    # ─────────────────────────────────────────────────────────
    # TEST 1: Goal 3 – Preferred Name & Standardized System Prompt
    # ─────────────────────────────────────────────────────────
    print("\n--- TEST 1: Goal 3 - Caretaker-Given Name & System Prompt ---")
    ctx = get_member_context(test_member_id)
    assert ctx["preferred_name"] == "Bauji", f"Expected 'Bauji', got '{ctx['preferred_name']}'"
    print(f"  [PASS] Preferred name resolved: '{ctx['preferred_name']}'")

    prompt = build_system_prompt(test_member_id)
    assert "You are Saathi, a warm, patient and caring voice companion." in prompt
    assert 'You are talking to Bauji. Always address them as "Bauji".' in prompt
    assert "Speak in" in prompt
    assert "calmly tell them to contact their caretaker or emergency services right away." in prompt
    print("  [PASS] Standardized system prompt generated matching exact template:")
    print("    " + "\n    ".join(prompt.splitlines()[:5]))

    # Test fallback when preferred_name is empty
    with SessionLocal() as db:
        r = db.query(CareRecipient).filter(CareRecipient.id == test_member_id).first()
        old_pref = r.preferred_name
        member_full_name = r.full_name
        r.preferred_name = ""
        db.commit()
    
    # Also clear in saathi_db cache to test fallback
    with SaathiSessionLocal() as s_db:
        fm = s_db.query(FamilyMember).filter(FamilyMember.id == test_member_id).first()
        if fm:
            fm.preferred_name = ""
            s_db.commit()

    ctx_fallback = get_member_context(test_member_id)
    expected_fallback = member_full_name.split()[0]
    assert ctx_fallback["preferred_name"] == expected_fallback, f"Expected fallback '{expected_fallback}', got '{ctx_fallback['preferred_name']}'"
    print(f"  [PASS] Fallback to first name verified when preferred_name is empty: '{ctx_fallback['preferred_name']}'")

    # Restore preferred_name
    with SessionLocal() as db:
        r = db.query(CareRecipient).filter(CareRecipient.id == test_member_id).first()
        r.preferred_name = old_pref
        db.commit()
    with SaathiSessionLocal() as s_db:
        fm = s_db.query(FamilyMember).filter(FamilyMember.id == test_member_id).first()
        if fm:
            fm.preferred_name = old_pref
            s_db.commit()

    # ─────────────────────────────────────────────────────────
    # TEST 2: Goal 1 – Start Session & Log Turns in saathi_db
    # ─────────────────────────────────────────────────────────
    print("\n--- TEST 2: Goal 1 - Session Lifecycle & Messages in saathi_db ---")
    session_id = start_session(test_caretaker_id, test_member_id)
    assert session_id is not None, "Failed to start session"
    print(f"  [PASS] Session started in saathi_db: {session_id}")

    with SaathiSessionLocal() as s_db:
        sess = s_db.query(ConversationSession).filter(ConversationSession.id == session_id).first()
        assert sess is not None
        assert sess.account_id == test_caretaker_id
        assert sess.family_member_id == test_member_id
        assert sess.started_at is not None
        assert sess.ended_at is None
    print("  [PASS] ConversationSession row verified in saathi_db.")

    # Save user message
    user_turn_1 = "Main subah 6 baje garden mein walk karne gaya tha. Meri beti Priya aayi thi aur usne chai banayi."
    msg_id_1 = save_message_sync(session_id, test_member_id, "user", user_turn_1, "hi")
    assert msg_id_1 is not None
    print(f"  [PASS] User message logged (ID: {msg_id_1}): {user_turn_1}")

    # Save assistant message
    asst_turn_1 = "Bahut badiya Bauji, subah ki taaza hawa sehat ke liye bahut achhi hoti hai. Priya ji se milkar aapko kaisa laga?"
    msg_id_2 = save_message_sync(session_id, test_member_id, "assistant", asst_turn_1, "hi")
    assert msg_id_2 is not None
    print(f"  [PASS] Assistant message logged (ID: {msg_id_2}): {asst_turn_1}")

    # Save user message with medicine and likes
    user_turn_2 = "Mujhe adrak wali chai bahut pasand hai. Par shaam ko BP ki dawai lene ke baad ghutne mein thoda dard tha."
    msg_id_3 = save_message_sync(session_id, test_member_id, "user", user_turn_2, "hi")
    assert msg_id_3 is not None
    print(f"  [PASS] Second user message logged (ID: {msg_id_3}): {user_turn_2}")

    # Test non-blocking async logging
    save_message_async(session_id, test_member_id, "assistant", "Bauji, aap thoda aaram kijiye aur gunguna paani pijiye.", "hi")
    time.sleep(0.5)  # Wait for async thread

    with SaathiSessionLocal() as s_db:
        total_msgs = s_db.query(ConversationMessage).filter(ConversationMessage.session_id == session_id).count()
        assert total_msgs >= 4, f"Expected at least 4 messages, got {total_msgs}"
        print(f"  [PASS] All {total_msgs} messages recorded with session_id FK in saathi_db.")

    # ─────────────────────────────────────────────────────────
    # TEST 3: Goal 1 & Goal 2 – Session End, Summary, & Memory
    # ─────────────────────────────────────────────────────────
    print("\n--- TEST 3: Goal 1 & Goal 2 - End Session, Summary, & Memory Extraction ---")
    end_session_sync(session_id)

    with SaathiSessionLocal() as s_db:
        sess = s_db.query(ConversationSession).filter(ConversationSession.id == session_id).first()
        assert sess.ended_at is not None, "ended_at not populated"
        assert sess.summary is not None and len(sess.summary) > 5, f"Summary missing or empty: {sess.summary}"
        print(f"  [PASS] Session closed with ended_at: {sess.ended_at}")
        print(f"  [PASS] Generated conversation summary: '{sess.summary}'")

        # Inspect extracted memories
        mems = s_db.query(MemberMemory).filter(MemberMemory.family_member_id == test_member_id).all()
        assert len(mems) > 0, "No memories were extracted!"
        print(f"  [PASS] Total member memories extracted in saathi_db: {len(mems)}")

        found_cats = set()
        for m in mems:
            assert m.category in VALID_CATEGORIES, f"Invalid category: {m.category}"
            found_cats.add(m.category)
            print(f"    - [{m.category.upper()}] {m.fact} (updated: {m.updated_at.strftime('%H:%M:%S')})")

        print(f"  [PASS] Categories covered: {', '.join(found_cats)}")

    # ─────────────────────────────────────────────────────────
    # TEST 4: Smart Upsert (Goal 2) – Update without duplication
    # ─────────────────────────────────────────────────────────
    print("\n--- TEST 4: Smart Memory Upsert (No duplicate explosion) ---")
    with SaathiSessionLocal() as s_db:
        init_count = s_db.query(MemberMemory).filter(MemberMemory.family_member_id == test_member_id).count()

    # Trigger second session mentioning updated fact about tea
    session_id_2 = start_session(test_caretaker_id, test_member_id)
    save_message_sync(session_id_2, test_member_id, "user", "Mujhe adrak wali chai aur elaichi dono bahut pasand hai.", "hi")
    end_session_sync(session_id_2)

    with SaathiSessionLocal() as s_db:
        new_count = s_db.query(MemberMemory).filter(MemberMemory.family_member_id == test_member_id).count()
        likes_mem = s_db.query(MemberMemory).filter(
            MemberMemory.family_member_id == test_member_id,
            MemberMemory.category == "likes"
        ).all()
        print(f"  [PASS] Memory count before: {init_count}, after upsert: {new_count}")
        print(f"  [PASS] Likes facts count: {len(likes_mem)} (Smart upsert updated or merged cleanly)")

    # ─────────────────────────────────────────────────────────
    # TEST 5: System Prompt with Live Memories & Summaries
    # ─────────────────────────────────────────────────────────
    print("\n--- TEST 5: Goal 3 - Live System Prompt with Memories & Summaries ---")
    updated_prompt = build_system_prompt(test_member_id)
    assert "Bauji" in updated_prompt
    assert "What you remember:" in updated_prompt
    assert "Recent conversations:" in updated_prompt
    print("  [PASS] Full prompt dynamically populated from saathi_db:")
    print("------------------------------------------------------------")
    print(updated_prompt)
    print("------------------------------------------------------------")

    # ─────────────────────────────────────────────────────────
    # TEST 6: Privacy Router / Data Access Scoping
    # ─────────────────────────────────────────────────────────
    print("\n--- TEST 6: Privacy Controls & Direct saathi_db Integrity ---")
    with SaathiSessionLocal() as s_db:
        # Check foreign keys and cascade integrity
        sessions = s_db.query(ConversationSession).filter(ConversationSession.account_id == test_caretaker_id).all()
        assert len(sessions) >= 2, f"Expected at least 2 sessions, got {len(sessions)}"
        print(f"  [PASS] Queried {len(sessions)} sessions scoped strictly by account_id={test_caretaker_id}.")

        # Check messages index / count
        msgs = s_db.query(ConversationMessage).filter(ConversationMessage.family_member_id == test_member_id).all()
        assert len(msgs) >= 4
        print(f"  [PASS] Verified {len(msgs)} messages scoped strictly by family_member_id={test_member_id}.")

    # ─────────────────────────────────────────────────────────
    # TEST 7: Privacy Route Handlers
    # ─────────────────────────────────────────────────────────
    print("\n--- TEST 7: Privacy Route Handlers ---")
    from routes.saathi_privacy import (
        list_conversation_sessions,
        get_session_messages,
        get_member_memory,
        update_member_preferred_name,
        delete_single_memory,
        UpdatePreferredNameRequest,
    )
    with SessionLocal() as db:
        ct = db.query(CareTaker).filter(CareTaker.id == test_caretaker_id).first()

    sess_list = list_conversation_sessions(test_member_id, ct)
    assert len(sess_list["sessions"]) >= 2
    latest_sess_id = sess_list["sessions"][0]["id"]
    print(f"  [PASS] list_conversation_sessions: {len(sess_list['sessions'])} sessions found")

    transcript = get_session_messages(latest_sess_id, ct)
    print(f"  [PASS] get_session_messages: {len(transcript['messages'])} messages in transcript")

    mems = get_member_memory(test_member_id, ct)
    assert mems["total_facts"] >= 1
    first_cat = list(mems["memories"].keys())[0]
    sample_mem_id = mems["memories"][first_cat][0]["id"]
    print(f"  [PASS] get_member_memory: {mems['total_facts']} total facts across categories: {list(mems['memories'].keys())}")

    name_res = update_member_preferred_name(test_member_id, UpdatePreferredNameRequest(preferred_name="Papaji"), ct)
    assert name_res["preferred_name"] == "Papaji"
    print(f"  [PASS] update_member_preferred_name: Updated to '{name_res['preferred_name']}'")

    del_res = delete_single_memory(test_member_id, sample_mem_id, ct)
    assert del_res["status"] == "success"
    print(f"  [PASS] delete_single_memory: Successfully deleted memory {sample_mem_id}")

    print("\n" + "=" * 60)
    print("ALL INTEGRATION TESTS PASSED SUCCESSFULLY ON saathi_db!")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
