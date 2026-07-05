import requests
from engine.prompts import SYSTEM_PROMPT
from config import OLLAMA_BASE_URL, OLLAMA_MODEL

def build_prompt(user_input, context=""):
    """Constructs the full prompt for the LLM."""
    return f"{SYSTEM_PROMPT}\n\nContext: {context}\n\nUser: {user_input}"

def persona_guard(response):
    """Ensures the response maintains the CriderGPT persona."""
    if "CriderGPT" not in response:
        return response + "\n\n(Response verified by CriderGPT persona guard)"
    return response

def generate(user_input, context=""):
    """Generates a response using Ollama and applies the persona guard."""
    prompt = build_prompt(user_input, context)
    
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }
    
    try:
        response = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload)
        response.raise_for_status()
        data = response.json()
        raw_response = data.get("response", "")
        return persona_guard(raw_response)
    except Exception as e:
        return f"Error during inference: {str(e)}"
