from __future__ import annotations

from fastapi import Request

from backend.app.core.errors import AppError
from backend.app.services.auth_service import UserRecord
from backend.app.services.container import ServiceContainer


def get_services(request: Request) -> ServiceContainer:
    return request.app.state.services


def get_client_ip(request: Request) -> str:
    client = request.client
    return client.host if client else "unknown"


def get_bearer_token(request: Request) -> str:
    authorization = request.headers.get("Authorization", "")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise AppError(401, "Missing bearer token.", "AUTH_BEARER_MISSING")
    return token


def require_user(request: Request) -> UserRecord:
    services = get_services(request)
    token = get_bearer_token(request)
    return services.auth_service.authenticate_token(token)


def require_admin(request: Request) -> UserRecord:
    user = require_user(request)
    if user.role != "admin":
        raise AppError(403, "Administrator role required.", "AUTH_ADMIN_REQUIRED")
    return user