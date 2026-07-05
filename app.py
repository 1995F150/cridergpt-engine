from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from engine.agent import get_agent_response
from memory.memory_exporter import export_memory

app = FastAPI(title="CriderGPT Engine")

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

@app.post("/sync")
async def sync_memory():
    """Sync memory to disk."""
    try:
        export_memory()
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
