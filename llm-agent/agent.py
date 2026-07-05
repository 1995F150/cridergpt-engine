from transformers import pipeline
from .memory_store import get_formatted_context
from .config import MODEL_NAME

def get_agent_response(user_input):
    """Uses the local memory context to answer user queries."""
    # Load the LLM pipeline
    generator = pipeline("text-generation", model=MODEL_NAME)
    
    # Get local memory context from memory_store
    context = get_formatted_context()
    
    # Construct the prompt with memory context
    prompt = f"Agent Memory Context:\n{context}\n\nUser Question: {user_input}\nAgent Response:"
    
    # Generate response
    result = generator(prompt, max_length=200, num_return_sequences=1, truncation=True)
    
    return result[0]['generated_text']

if __name__ == "__main__":
    test_input = "What information do you have in your memory?"
    print(f"User: {test_input}")
    print(f"Agent: {get_agent_response(test_input)}")
