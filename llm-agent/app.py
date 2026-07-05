from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .agent import get_agent_response
from .memory_exporter import export_memory

app = FastAPI(title="LLM Agent Backend")

class Query(BaseModel):
    text: str

@app.post("/ask")
async def ask_agent(query: Query):
    """Endpoint for the website/app to talk to the agent."""
    try:
        response = get_agent_response(query.text)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/sync-memory")
async def sync_memory():
    """Manual trigger to pull from Supabase and update local files."""
    try:
        export_memory()
        return {"status": "Memory synced successfully from Supabase to local files."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "LLM Agent API is running."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
