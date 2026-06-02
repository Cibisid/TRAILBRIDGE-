"""
TrialBridge — Security & JWT Authentication

Why JWT? It's stateless — the server doesn't need to store sessions.
Each token is self-contained and cryptographically signed.
The server only needs the secret key to validate any token.
This scales horizontally — any server instance can validate any token.
"""

from datetime import UTC, datetime, timedelta
from typing import Any

import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from pydantic import BaseModel

from backend.core.config import get_settings
from backend.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

# FastAPI security scheme — reads Bearer token from Authorization header
bearer_scheme = HTTPBearer()


# -----------------------------------------------
# Token Models
# -----------------------------------------------
class TokenPayload(BaseModel):
    """What we store inside the JWT token."""
    sub: str           # subject — the user ID
    exp: datetime      # expiry timestamp
    iat: datetime      # issued at timestamp
    token_type: str    # "access" or "refresh"
    scopes: list[str]  # permissions: ["match:read", "match:write"]


class TokenResponse(BaseModel):
    """What we return to the client after login."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds until access token expires


# -----------------------------------------------
# Password Hashing
# -----------------------------------------------
def hash_password(plain_password: str) -> str:
    """
    Hash a password using bcrypt.
    bcrypt is slow by design — it makes brute-force attacks impractical.
    Never store plain passwords. Ever.
    """
    salt = bcrypt.gensalt(rounds=12)  # 12 rounds = ~250ms per hash
    return bcrypt.hashpw(plain_password.encode(), salt).decode()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain password against its bcrypt hash."""
    return bcrypt.checkpw(
        plain_password.encode(),
        hashed_password.encode(),
    )


# -----------------------------------------------
# JWT Token Creation
# -----------------------------------------------
def create_access_token(
    user_id: str,
    scopes: list[str] | None = None,
    extra_data: dict[str, Any] | None = None,
) -> str:
    """
    Create a short-lived JWT access token.
    Default expiry: 60 minutes (configurable via env).
    """
    now = datetime.now(UTC)
    expire = now + timedelta(minutes=settings.jwt_access_token_expire_minutes)

    payload = {
        "sub": user_id,
        "exp": expire,
        "iat": now,
        "token_type": "access",
        "scopes": scopes or ["match:read"],
        **(extra_data or {}),
    }

    return jwt.encode(
        payload,
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
    )


def create_refresh_token(user_id: str) -> str:
    """
    Create a long-lived refresh token.
    Used to get a new access token without re-logging in.
    Default expiry: 7 days.
    """
    now = datetime.now(UTC)
    expire = now + timedelta(days=settings.jwt_refresh_token_expire_days)

    payload = {
        "sub": user_id,
        "exp": expire,
        "iat": now,
        "token_type": "refresh",
        "scopes": [],
    }

    return jwt.encode(
        payload,
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
    )


def create_token_pair(user_id: str, scopes: list[str] | None = None) -> TokenResponse:
    """Create both access and refresh tokens for a user."""
    access_token = create_access_token(user_id, scopes)
    refresh_token = create_refresh_token(user_id)

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=settings.jwt_access_token_expire_minutes * 60,
    )


# -----------------------------------------------
# JWT Token Validation
# -----------------------------------------------
def decode_token(token: str) -> TokenPayload:
    """
    Decode and validate a JWT token.
    Raises HTTPException if token is invalid or expired.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
        )
        return TokenPayload(**payload)

    except JWTError as e:
        logger.warning("JWT validation failed", error=str(e))
        raise credentials_exception


# -----------------------------------------------
# FastAPI Dependencies
# -----------------------------------------------
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> TokenPayload:
    """
    FastAPI dependency — extracts and validates the JWT from the request.
    Add to any route that requires authentication:

        @router.get("/match")
        async def match(user: TokenPayload = Depends(get_current_user)):
            ...
    """
    return decode_token(credentials.credentials)


def require_scope(scope: str):
    """
    FastAPI dependency factory — ensures the user has a specific permission scope.
    Usage:
        @router.post("/admin/ingest")
        async def ingest(user = Depends(require_scope("admin:write"))):
            ...
    """
    async def _check_scope(
        user: TokenPayload = Depends(get_current_user),
    ) -> TokenPayload:
        if scope not in user.scopes:
            logger.warning(
                "Insufficient scope",
                user_id=user.sub,
                required=scope,
                has=user.scopes,
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required scope: {scope}",
            )
        return user

    return _check_scope
