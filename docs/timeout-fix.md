# Engine timeout fix

The Supabase Edge Function aborts its engine request after 120 seconds. The engine now:

- limits duplicated system-prompt and memory context,
- limits conversation-history size,
- converts cloud gateway model IDs to the configured local Ollama model,
- caps Ollama generation at 90 seconds,
- returns a controlled 503 before Supabase terminates the request.

This addresses the observed `engine unreachable: Signal timed out` failure at the engine layer. A cloud fallback in the Edge Function can be added separately if engine-only operation is no longer desired.
