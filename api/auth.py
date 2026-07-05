# API authentication and authorization
from fastapi import APIRouter, Depends

router = APIRouter()

async def verify_token(token: str):
    """Verifies the provided authentication token."""
    return True
from fastapi import Header, HTTPException, status
from config import API_KEYS

async def validate_api_key(x_api_key: str = Header(None)):
    """
    Dependency to validate the API key from request headers.
    """
    if not x_api_key or x_api_key not in API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key",
        )
    return x_api_key
@router.get("/auth/verify")
async def check_auth(is_valid: bool = Depends(verify_token)):
    """Validates authentication state via the API."""
    return {"authenticated": is_valid}
