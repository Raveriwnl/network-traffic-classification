from __future__ import annotations

from fastapi import APIRouter, Query, Request

from backend.app.api.deps import get_services, require_admin
from backend.app.schemas.flows import FlowDetailResponse, FlowListResponse, PredictionsLatestResponse


router = APIRouter(tags=["flows"])


@router.get("/api/flows", response_model=FlowListResponse)
def list_flows(
    request: Request,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=200),
    source_ip: str | None = Query(default=None),
    dest_ip: str | None = Query(default=None),
    protocol: str | None = Query(default=None),
    eligible_for_inference: bool | None = Query(default=None),
    predicted_class: str | None = Query(default=None),
    start_time: str | None = Query(default=None),
    end_time: str | None = Query(default=None),
) -> dict[str, object]:
    require_admin(request)
    services = get_services(request)
    return services.flow_service.list_flows(
        page=page,
        page_size=page_size,
        source_ip=source_ip,
        dest_ip=dest_ip,
        protocol=protocol,
        eligible_for_inference=eligible_for_inference,
        predicted_class=predicted_class,
        start_time=start_time,
        end_time=end_time,
    )


@router.get("/api/flows/{flow_id}", response_model=FlowDetailResponse)
def get_flow_detail(flow_id: int, request: Request) -> dict[str, object]:
    require_admin(request)
    services = get_services(request)
    return services.flow_service.get_flow_detail(flow_id)


@router.get("/api/predictions/latest", response_model=PredictionsLatestResponse)
def latest_predictions(request: Request, limit: int = Query(default=20, ge=1, le=200)) -> dict[str, object]:
    require_admin(request)
    services = get_services(request)
    return services.telemetry_service.list_recent_predictions(limit)