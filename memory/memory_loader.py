

def get_conversation_history():
    """Retrieves and formats the recent conversation history as a string."""
    if not supabase:
        return ""
    try:
        # Fetch last 10 messages
        response = supabase.table("chat_messages").select("role, content").order("created_at", descending=True).limit(10).execute()
        messages = response.data[::-1]
        
        history_str = ""
        for msg in messages:
            role = msg.get("role", "unknown").capitalize()
            content = msg.get("content", "")
            history_str += f"{role}: {content}\n"
        return history_str.strip()
    except Exception:
        return ""
