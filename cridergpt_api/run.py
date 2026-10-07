"""Launch CriderGPT API."""
import os
import uvicorn
if __name__=="__main__":
    uvicorn.run("cridergpt_api.server:app",host=os.getenv("CRIDERGPT_HOST","127.0.0.1"),port=int(os.getenv("CRIDERGPT_PORT","8000")),workers=1)
