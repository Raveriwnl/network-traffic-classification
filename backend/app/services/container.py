from __future__ import annotations

from dataclasses import dataclass

from backend.app.core.config import Settings
from backend.app.services.audit_service import AuditService
from backend.app.services.auth_service import AuthService
from backend.app.services.collector_service import CollectorService
from backend.app.services.database_service import DatabaseService
from backend.app.services.flow_service import FlowService
from backend.app.services.model_service import ModelService
from backend.app.services.runtime_service import RuntimeService
from backend.app.services.telemetry_service import TelemetryService


@dataclass(slots=True)
class ServiceContainer:
    settings: Settings
    database_service: DatabaseService
    audit_service: AuditService
    auth_service: AuthService
    collector_service: CollectorService
    flow_service: FlowService
    model_service: ModelService
    telemetry_service: TelemetryService
    runtime_service: RuntimeService