"""API keys (§ARCH 7.16, 9): `X-API-Key` compared in constant time (`hmac.compare_digest`).

The user key opens the user endpoints; the admin key opens those and `/v1/documents`. A missing
or unknown key is 401; the user key on an admin endpoint is 403.
"""

from __future__ import annotations

import hmac
from typing import Literal

from fastapi import Request
from pydantic import SecretStr

from credagent.api.errors import ApiError
from credagent.settings import Settings

Role = Literal["user", "admin"]


def _matches(given: str, expected: SecretStr) -> bool:
    return hmac.compare_digest(given.encode("utf-8"), expected.get_secret_value().encode("utf-8"))


def role_for(given: str | None, settings: Settings) -> Role | None:
    if given is None:
        return None
    admin = _matches(given, settings.admin_api_key)  # both compared: no early exit on a match
    user = _matches(given, settings.api_key)
    return "admin" if admin else "user" if user else None


async def require_user(request: Request) -> Role:
    role = role_for(request.headers.get("x-api-key"), request.app.state.settings)
    if role is None:
        raise ApiError("UNAUTHORIZED")
    return role


async def require_admin(request: Request) -> Role:
    role = await require_user(request)
    if role != "admin":
        raise ApiError("FORBIDDEN")
    return role
