import json
import os
from .config import DATA_DIR

def load_local_memory():
    """Loads the local memory files and formats them into useful context."""
    json_path = os.path.join(DATA_DIR, "ai_memory.json")

    if not os.path.exists(json_path):
        return "No local memory found."

    try:
        with open(json_path, "r") as f:
            memories = json.load(f)

        # Format into a simple context string
        context = "\n".join([m.get("content", "") for m in memories[-5:]])
        return context
    except Exception as e:
        return f"Error loading memory: {str(e)}"

def get_formatted_context():
    """Returns memory formatted for the agent."""
    return load_local_memory()
