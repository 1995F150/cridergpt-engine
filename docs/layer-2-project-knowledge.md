# Layer 2: Project Knowledge

Layer 2 stores durable, user-scoped project information separately from short-term chat memory.

## Data model

Each row belongs to one authenticated user and includes:

- project key and display name
- category and title
- durable content
- status and priority
- source and metadata
- active/inactive state

## Retrieval

The engine retrieves up to eight active entries, ranks them by project-name, heading, and body matches, and inserts them immediately after the core profile in the system context.

This is a bounded keyword retrieval layer. Semantic vector retrieval will replace or augment the scorer in Layer 4.

## Applying the migration

Apply `supabase/migrations/20260727_create_project_knowledge.sql` to the same Supabase project used by the engine.

## Timeout hardening

This layer also reduces oversized prompt duplication, ignores cloud gateway model IDs passed to Ollama, caps Ollama processing below the Edge Function deadline, and returns a controlled engine error before Supabase terminates the request.
