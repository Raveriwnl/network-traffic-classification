from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
import sys
from uuid import uuid4


if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.api import admin, auth, flows, telemetry
from backend.app.core.config import Settings
from backend.app.core.errors import AppError, build_error_payload
from backend.app.services.audit_service import AuditService
from backend.app.services.auth_service import AuthService
from backend.app.services.collector_service import CollectorService
from backend.app.services.container import ServiceContainer
from backend.app.services.database_service import DatabaseService
from backend.app.services.flow_service import FlowService
from backend.app.services.model_service import ModelService
from backend.app.services.runtime_service import RuntimeService
from backend.app.services.telemetry_service import TelemetryService


def build_services() -> ServiceContainer:
    settings = Settings.from_env()
    database_service = DatabaseService(settings)
    audit_service = AuditService(database_service=database_service)
    flow_service = FlowService(settings, database_service)
    model_service = ModelService(settings, flow_service.class_names)
    collector_service = CollectorService(settings, audit_service, database_service, model_service)
    telemetry_service = TelemetryService(settings, flow_service, database_service)
    runtime_service = RuntimeService(
        telemetry_service,
        audit_service,
        database_service,
        collector_service,
        settings.telemetry_interval_sec,
    )
    auth_service = AuthService(settings, audit_service, database_service)
    audit_service.append(
        level="info",
        action="service_boot",
        message="Backend service initialized.",
        actor="system",
        role="service",
        ip="127.0.0.1",
    )
    return ServiceContainer(
        settings=settings,
        database_service=database_service,
        audit_service=audit_service,
        auth_service=auth_service,
        collector_service=collector_service,
        flow_service=flow_service,
        model_service=model_service,
        telemetry_service=telemetry_service,
        runtime_service=runtime_service,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    services = build_services()
    app.state.services = services
    await services.runtime_service.start()
    try:
        yield
    finally:
        await services.runtime_service.stop()
        services.database_service.close()


app = FastAPI(title="Edge Traffic Classification Backend", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(Settings.from_env().frontend_origins),
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1|0\.0\.0\.0|192\.168\.\d+\.\d+|10\.\d+\.\d+\.\d+|172\.(1[6-9]|2\d|3[0-1])\.\d+\.\d+)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def attach_request_id(request: Request, call_next):
    request_id = uuid4().hex
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    request_id = getattr(request.state, "request_id", uuid4().hex)
    return JSONResponse(
        status_code=exc.status_code,
        content=build_error_payload(exc.detail, exc.code, request_id),
    )


@app.exception_handler(HTTPException)
async def http_error_handler(request: Request, exc: HTTPException) -> JSONResponse:
    request_id = getattr(request.state, "request_id", uuid4().hex)
    detail = exc.detail if isinstance(exc.detail, str) else "Request failed."
    return JSONResponse(
        status_code=exc.status_code,
        content=build_error_payload(detail, "HTTP_ERROR", request_id),
    )


app.include_router(auth.router)
app.include_router(telemetry.router)
app.include_router(admin.router)
app.include_router(flows.router)


@app.websocket("/ws/telemetry")
async def telemetry_websocket(websocket: WebSocket) -> None:
    services: ServiceContainer = websocket.app.state.services
    token = websocket.query_params.get("token")
    if not token:
        await websocket.accept()
        await websocket.close(code=4401, reason="Missing token.")
        return
    try:
        user = services.auth_service.authenticate_token(token)
    except AppError:
        await websocket.accept()
        await websocket.close(code=4401, reason="Invalid token.")
        return
    if user.role not in {"admin", "analyst"}:
        await websocket.accept()
        await websocket.close(code=4403, reason="Forbidden.")
        return
    await websocket.accept()
    client = websocket.client.host if websocket.client else "unknown"
    services.audit_service.append(
        level="info",
        action="ws_connected",
        message="Authenticated telemetry websocket connected.",
        actor=user.username,
        role=user.role,
        ip=client,
    )
    version, payload = services.runtime_service.get_latest()
    try:
        await websocket.send_json(payload)
        while True:
            version, payload = await services.runtime_service.wait_for_update(version, timeout_sec=25.0)
            await websocket.send_json(payload)
    except WebSocketDisconnect:
        services.audit_service.append(
            level="warn",
            action="ws_disconnected",
            message="Telemetry websocket disconnected.",
            actor=user.username,
            role=user.role,
            ip=client,
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)