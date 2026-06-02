-- TrialBridge — PostgreSQL Initialization
-- Runs once when the container is first created.

-- Enable pgvector extension for semantic similarity search
CREATE EXTENSION IF NOT EXISTS vector;

-- Enable pg_trgm for fast text search (trial titles, conditions)
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Enable uuid-ossp for UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Log the setup
DO $$
BEGIN
    RAISE NOTICE 'TrialBridge database initialized with pgvector, pg_trgm, uuid-ossp';
END $$;
