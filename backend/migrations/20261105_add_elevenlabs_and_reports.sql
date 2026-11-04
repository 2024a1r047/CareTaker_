-- Migration: 20261105_add_elevenlabs_and_reports.sql
-- Database: saathi_db

-- Add elevenlabs_conversation_id to conversation_sessions
ALTER TABLE conversation_sessions 
ADD COLUMN IF NOT EXISTS elevenlabs_conversation_id TEXT UNIQUE;

CREATE INDEX IF NOT EXISTS idx_conv_sessions_elevenlabs_id 
ON conversation_sessions(elevenlabs_conversation_id);

-- Add source_session_id to member_memory
ALTER TABLE member_memory 
ADD COLUMN IF NOT EXISTS source_session_id UUID REFERENCES conversation_sessions(id) ON DELETE SET NULL;
