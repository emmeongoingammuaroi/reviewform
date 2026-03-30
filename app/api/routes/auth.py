"""Auth routes — simple JWT token generation for development."""

from datetime import UTC, datetime, timedelta

from fastapi import APIRouter
from jose import jwt
from pydantic import BaseModel

from app.core.config import settings

router = APIRouter(prefix="/auth", tags=["auth"])


class TokenRequest(BaseModel):
    user_id: str
    email: str = ""


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


@router.post("/token", response_model=TokenResponse)
async def create_token(request: TokenRequest) -> TokenResponse:
    """Generate a JWT token for development/testing.

    In production, you'd integrate with OAuth2 / SSO instead.
    """
    expire = datetime.now(UTC) + timedelta(minutes=settings.jwt_expire_minutes)
    payload = {
        "sub": request.user_id,
        "email": request.email,
        "exp": expire,
    }
    token = jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)

    return TokenResponse(access_token=token)
