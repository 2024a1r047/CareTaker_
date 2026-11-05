-- Migration: create_saathi_db_tables.sql
-- Database: saathi_db

-- Enable UUID extension if needed
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- 1. Family Members Table (mirrors / stores family member profiles scoped to account_id)
CREATE TABLE IF NOT EXISTS family_members (
    id BIGINT PRIMARY KEY,
    account_id BIGINT NOT NULL,
    full_name VARCHAR(255) NOT NULL,
    preferred_name VARCHAR(255),
    relationship VARCHAR(100),
    language VARCHAR(50) DEFAULT 'hi',
    profile_details JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_family_members_account ON family_members(account_id);

-- 2. Conversation Sessions Table
CREATE TABLE IF NOT EXISTS member_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    account_id BIGINT NOT NULL,
    family_member_id BIGINT NOT NULL REFERENCES family_members(id) ON DELETE CASCADE,
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    ended_at TIMESTAMPTZ,
    summary TEXT
);
CREATE INDEX IF NOT EXISTS idx_member_sessions_member ON member_sessions(family_member_id, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_member_sessions_account ON member_sessions(account_id);

-- 3. Conversation Messages Table
CREATE TABLE IF NOT EXISTS member_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES member_sessions(id) ON DELETE CASCADE,
    family_member_id BIGINT NOT NULL REFERENCES family_members(id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    language VARCHAR(20) DEFAULT 'hi',
    audio_url TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_member_messages_member_created ON member_messages(family_member_id, created_at);

-- 4. Member Personal Profile Memory Table
CREATE TABLE IF NOT EXISTS member_memory (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    family_member_id BIGINT NOT NULL REFERENCES family_members(id) ON DELETE CASCADE,
    category TEXT NOT NULL CHECK (category IN ('health', 'medicine', 'family', 'likes', 'dislikes', 'routine', 'mood')),
    fact TEXT NOT NULL,
    source_message_id UUID REFERENCES member_messages(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_member_memory_member_cat ON member_memory(family_member_id, category);

