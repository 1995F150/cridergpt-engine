import json
import os
try:
    from config import DATA_DIR
except ImportError:
    DATA_DIR = "data"

def load_local_memory():
    """Loads the local memory files and formats them inimport os
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

def load_local_memory():
    """Loads the local memory files and formats them into useful context."""
    json_path = os.path.join(DATA_DIR, "ai_memory.json")
    if not os.path.exists(json_path):
        return "No local memory found."
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except Exception:
        return "Error loading local memory."to useful context."""
    json_path = os.path.join(DATA_DIR, "ai_memory.json")

    if not os.path.exists(json_path):
        return "No local memory found."

    try:
        with open(json_path, "r") as f:
            memories = json.load(f)

        # Format into a simple context string
        context = "\n".join([m.get("content", "") for m in memories])
        return context
    except Exception as e:
        return f"Error loading memory: {str(e)}"

def get_formatted_context():
    """Returns memory formatted for the agent."""
    return load_local_memory()
