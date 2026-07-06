import os
import json
from supabase import create_client, Client

try:
    from config import SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, DATA_DIR
except ImportError:
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    DATA_DIR = "data"

supabase: Client = None

def init_supabase():
    global supabase
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        print("Supabase credentials missing. Degrading gracefully.")
        return None
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
        return supabase
    except Exception as e:
        print(f"Failed to initialize Supabase: {e}. Degrading gracefully.")
        return None

# Initialize on module load
init_supabase()

def get_formatted_context(user_id):
    """Loads the local memory files and formats them into useful context."""
    # Add your local memory fallback logic here if Supabase degrades
    return ""
