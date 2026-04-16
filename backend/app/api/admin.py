from __future__ import annotations

from fastapi import APIRouter, Query, Request

from backend.app.api.deps import get_client_ip, get_services, require_admin
from backend.app.core.errors import AppError
from backend.app.schemas.admin import CaptureSessionResponse, CaptureStartRequest, LogsResponse
from backend.app.schemas.auth import UserCreateRequest, UserItemResponse, UserUpdateRequest, UsersResponse


router = APIRouter(prefix="/api/admin", tags=["admin"])


@router.get("/logs", response_model=LogsResponse)
def admin_logs(
    request: Request,
    limit: int = Query(default=120, ge=1, le=200),
    level: str | None = Query(default=None),
    actor: str | None = Query(default=None),
    start_time: str | None = Query(default=None),
    end_time: str | None = Query(default=None),
) -> dict[str, object]:
    admin = require_admin(request)
    services = get_services(request)
    services.audit_service.append(
        level="info",
        action="admin_logs_view",
        message="Administrator opened the runtime log viewer.",
        actor=admin.username,
        role=admin.role,
        ip=get_client_ip(request),
    )
    return {
        "logs": services.audit_service.query(
            limit=limit,
            level=level,
            actor=actor,
            start_time=start_time,
            end_time=end_time,
        )
    }


@router.post("/capture/start", response_model=CaptureSessionResponse)
def capture_start(payload: CaptureStartRequest, request: Request) -> dict[str, object]:
    admin = require_admin(request)
    services = get_services(request)
    try:
        return services.runtime_service.start_capture(admin.username, admin.role, get_client_ip(request), payload)
    except ValueError as exc:
        raise AppError(409, str(exc), "CAPTURE_ALREADY_RUNNING") from exc


@router.post("/capture/stop", response_model=CaptureSessionResponse)
def capture_stop(request: Request) -> dict[str, object]:
    admin = require_admin(request)
    services = get_services(request)
    try:
        return services.runtime_service.stop_capture(admin.username, admin.role, get_client_ip(request))
    except ValueError as exc:
        raise AppError(409, str(exc), "CAPTURE_NOT_RUNNING") from exc
    except RuntimeError as exc:
        raise AppError(500, str(exc), "CAPTURE_STOP_FAILED") from exc


@router.get("/users", response_model=UsersResponse)
def list_users(request: Request) -> dict[str, object]:
    admin = require_admin(request)
    services = get_services(request)
    services.audit_service.append(
        level="info",
        action="admin_users_view",
        message="Administrator listed users.",
        actor=admin.username,
        role=admin.role,
        ip=get_client_ip(request),
    )
    return {"users": services.auth_service.list_users()}


@router.post("/users", response_model=UserItemResponse)
def create_user(payload: UserCreateRequest, request: Request) -> dict[str, object]:
    admin = require_admin(request)
    services = get_services(request)
    user = services.auth_service.create_user(
        username=payload.username,
        password=payload.password,
        display_name=payload.display_name,
        role=payload.role,
    )
    services.audit_service.append(
        level="info",
        action="admin_user_create",
        message="Administrator created a new user.",
        actor=admin.username,
        role=admin.role,
        ip=get_client_ip(request),
    )
    return user


@router.patch("/users/{user_id}", response_model=UserItemResponse)
def update_user(user_id: int, payload: UserUpdateRequest, request: Request) -> dict[str, object]:
    admin = require_admin(request)
    services = get_services(request)
    user = services.auth_service.update_user(
        user_id,
        display_name=payload.display_name,
        role=payload.role,
        status=payload.status,
    )
    services.audit_service.append(
        level="info",
        action="admin_user_update",
        message="Administrator updated user metadata.",
        actor=admin.username,
        role=admin.role,
        ip=get_client_ip(request),
    )
    return user