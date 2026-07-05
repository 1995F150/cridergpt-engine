from supabase import create_client, Client
try:
    from config import SUPABASE_URL, SUPABASE_KEY
except ImportError:
    SUPABASE_URL = "your_supabase_url"
    SUPABASE_KEY = "your_supabase_key"

def load_memory_from_supabase():
    """Reads all rows from Supabase ai_memory table."""
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    response = supabase.table("ai_memory").select("*").execute()
    return response.data

def get_memories(user_id, query, recent_k=5, relevant_k=5):
    """
    Retrieves the top k recent and top k relevant memories for the user.
    """
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    # Fetch top k recent memories
    recent_res = supabase.table("ai_memory") \
        .select("*") \
        .eq("user_id", user_id) \
        .order("created_at", descending=True) \
        .limit(recent_k) \
        .execute()
    
    # Fetch top k relevant memories (Mocking semantic search with a simple query for now)
    relevant_res = supabase.table("ai_memory") \
        .select("*") \
        .eq("user_id", user_id) \
        .ilike("content", f"%{query}%") \
        .limit(relevant_k) \
        .execute()
    
    # Combine and deduplicate by memory ID
    mem_map = {m['id']: m for m in (recent_res.data + relevant_res.data)}
    return list(mem_map.values())
