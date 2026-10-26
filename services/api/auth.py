"""Supabase JWT verification.

Uses the asymmetric flow (ES256/RS256 + JWKS), not the legacy shared HS256 secret.
Supabase's own docs recommend against the shared secret: a leak lets anyone forge tokens
for any user indefinitely, and rotating it means downtime. The JWKS endpoint supports
zero-downtime rotation and the private key never leaves Supabase.

PyJWKClient caches the fetched keys, so this costs one HTTP round trip at startup rather
than one per request.
"""
from __future__ import annotations

from functools import lru_cache

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from services.api.config import get_settings

bearer_scheme = HTTPBearer(auto_error=True)
ALLOWED_ALGORITHMS = {"ES256", "RS256"}


@lru_cache(maxsize=1)
def _jwks_client() -> jwt.PyJWKClient:
    return jwt.PyJWKClient(get_settings().jwks_url, cache_keys=True)


class CurrentUser:
    __slots__ = ("id", "email", "claims")

    def __init__(self, user_id: str, email: str | None, claims: dict):
        self.id = user_id
        self.email = email
        self.claims = claims


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> CurrentUser:
    token = credentials.credentials
    settings = get_settings()

    # DEV_MODE lets the API run against a local database without a Supabase project,
    # by treating the bearer token as a raw user id. It is refused whenever real
    # Supabase credentials are configured, so it cannot be left on by accident.
    if settings.dev_mode and not settings.service_role_key:
        return CurrentUser(token, None, {"sub": token, "dev": True})

    try:
        header = jwt.get_unverified_header(token)
        algorithm = header.get("alg")
        if algorithm not in ALLOWED_ALGORITHMS:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Unsupported token algorithm: {algorithm}",
            )
        signing_key = _jwks_client().get_signing_key_from_jwt(token).key
        claims = jwt.decode(
            token,
            signing_key,
            algorithms=[algorithm],
            audience="authenticated",
            options={"require": ["exp", "sub"]},
        )
    except HTTPException:
        raise
    except jwt.PyJWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    return CurrentUser(claims["sub"], claims.get("email"), claims)
