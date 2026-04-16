from __future__ import annotations

from fastapi import APIRouter, Request

from backend.app.api.deps import get_client_ip, get_services, require_user
from backend.app.schemas.telemetry import TelemetryResponse


router = APIRouter(tags=["telemetry"])


@router.get("/api/health")
def health(request: Request) -> dict[str, object]:
    services = get_services(request)
    return {
        "status": "ok",
        "service": "edge-traffic-backend",
        "auth": "jwt",
        "model_name": services.settings.model_name,
        "model_version": services.settings.model_version,
        "class_names": services.flow_service.class_names,
    }


@router.get("/api/telemetry/latest", response_model=TelemetryResponse)
def latest_telemetry(request: Request) -> dict[str, object]:
    user = require_user(request)
    services = get_services(request)
    services.audit_service.append(
        level="info",
        action="telemetry_snapshot",
        message="Frontend requested latest telemetry snapshot.",
        actor=user.username,
        role=user.role,
        ip=get_client_ip(request),
    )
    return services.telemetry_service.get_latest_snapshot()