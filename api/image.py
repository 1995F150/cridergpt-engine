from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from engine.helpers import resolve_reference_alias

router = APIRouter(prefix="/image", tags=["image"])

class ImageGenerationRequest(BaseModel):
    prompt: str
    character_name: str | None = None

class ImageAnalysisRequest(BaseModel):
    image_url: str

@router.post("/generate")
async def generate_image(request: ImageGenerationRequest):
    references = []
    if request.character_name:
        references = resolve_reference_alias(request.character_name)
    
    return {
        "status": "success", 
        "message": "Image generated successfully",
        "prompt": request.prompt, 
        "references": references
    }

@router.post("/analyze")
async def analyze_image(request: ImageAnalysisRequest):
    return {
        "status": "success", 
        "message": "Image analyzed successfully",
        "analysis": "This is a placeholder analysis for the image."
    }
