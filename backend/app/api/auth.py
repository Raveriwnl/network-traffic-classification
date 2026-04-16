from __future__ import annotations

from fastapi import APIRouter, Request

from backend.app.api.deps import get_client_ip, get_services, require_user
from backend.app.schemas.auth import LoginRequest, TokenResponse, UserResponse


router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest, request: Request) -> dict[str, object]:
    services = get_services(request)
    return services.auth_service.login(
        username=payload.username,
        password=payload.password,
        ip=get_client_ip(request),
        user_agent=request.headers.get("user-agent"),
    )


@router.get("/me", response_model=UserResponse)
def me(request: Request) -> dict[str, str]:
    user = require_user(request)
    services = get_services(request)
    services.audit_service.append(
        level="info",
        action="profile_check",
        message="Frontend refreshed active profile.",
        actor=user.username,
        role=user.role,
        ip=get_client_ip(request),
    )
    return user.public_dict()