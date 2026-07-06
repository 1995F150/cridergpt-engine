from transformers import pipeline
from memory.memory_store import get_formatted_context
import logging

try:
    from config import MODEL_NAME
except ImportError:
    MODEL_NAME = "gpt2"

logger = logging.getLogger(__name__)

def get_agent_response(message: str, user_id: str, conversation_id: str = None) -> str:
    """Uses the local memory context to answer user queries."""
    # Load the LLM pipeline
    generator = pipeline("text-generation", model=MODEL_NAME)
    
    # Get local memory context from memory_store
    try:
        context = get_formatted_context(user_id=user_id)
    except Exception as e:
        logger.error(f"Supabase context fetch failed: {e}")
        context = ""

    # Construct the prompt with memory context
    prompt = f"Agent Memory Context:\n{context}\n\nUser Question: {message}\nAgent Response:"

    # Generate response
    result = generator(prompt, max_length=200, num_return_sequences=1, truncation=True)

    return result[0]['generated_text']

if __name__ == "__main__":
    test_input = "What information do you have in your memory?"
    print(f"User: {test_input}")
    print(f"Agent: {get_agent_response(test_input, user_id='test_user')}")
