from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from backend.app.schemas.admin import CaptureStartRequest
from backend.app.services.audit_service import AuditService
from backend.app.services.database_service import DatabaseService
from backend.app.services.collector_service import CollectorService
from backend.app.services.telemetry_service import TelemetryService


class RuntimeService:
    def __init__(
        self,
        telemetry_service: TelemetryService,
        audit_service: AuditService,
        database_service: DatabaseService,
        collector_service: CollectorService,
        interval_sec: float,
    ) -> None:
        self.telemetry_service = telemetry_service
        self.audit_service = audit_service
        self.database_service = database_service
        self.collector_service = collector_service
        self.interval_sec = interval_sec
        self._task: asyncio.Task[None] | None = None
        self._event = asyncio.Event()
        self._version = 0
        self._next_cleanup_at = datetime.now(timezone.utc).timestamp() + self.telemetry_service.settings.cleanup_interval_sec

    async def start(self) -> None:
        self.telemetry_service.get_latest_snapshot()
        settings = self.telemetry_service.settings
        if settings.auto_capture_enabled:
            try:
                self.collector_service.start_capture(
                    actor="system",
                    role="service",
                    ip="127.0.0.1",
                    iface=settings.auto_capture_iface,
                    bpf_filter=settings.auto_capture_bpf_filter,
                    idle_timeout=settings.auto_capture_idle_timeout,
                    min_packets=settings.auto_capture_min_packets,
                    capture_label=settings.auto_capture_label,
                )
            except ValueError as exc:
                self.audit_service.append(
                    level="warn",
                    action="capture_auto_start_skipped",
                    message=f"Automatic capture start skipped: {exc}",
                    actor="system",
                    role="service",
                    ip="127.0.0.1",
                )
            except Exception as exc:
                self.audit_service.append(
                    level="error",
                    action="capture_auto_start_failed",
                    message=f"Automatic capture start failed: {exc}",
                    actor="system",
                    role="service",
                    ip="127.0.0.1",
                )
        self._task = asyncio.create_task(self._telemetry_loop(), name="telemetry-loop")

    async def stop(self) -> None:
        self._run_cleanup()
        self.collector_service.shutdown()
        if not self._task:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass

    async def _telemetry_loop(self) -> None:
        while True:
            await asyncio.sleep(self.interval_sec)
            self.telemetry_service.generate_next_snapshot()
            self._version += 1
            self._event.set()
            self._event = asyncio.Event()
            now_ts = datetime.now(timezone.utc).timestamp()
            if now_ts >= self._next_cleanup_at:
                self._run_cleanup()
                self._next_cleanup_at = now_ts + self.telemetry_service.settings.cleanup_interval_sec

    def get_latest(self) -> tuple[int, dict[str, object]]:
        return self._version, self.telemetry_service.get_latest_snapshot()

    async def wait_for_update(self, version: int, timeout_sec: float) -> tuple[int, dict[str, object]]:
        if version != self._version:
            return self._version, self.telemetry_service.get_latest_snapshot()
        current_event = self._event
        try:
            await asyncio.wait_for(current_event.wait(), timeout=timeout_sec)
        except asyncio.TimeoutError:
            return self._version, self.telemetry_service.get_latest_snapshot()
        return self._version, self.telemetry_service.get_latest_snapshot()

    def start_capture(self, actor: str, role: str, ip: str, payload: CaptureStartRequest) -> dict[str, object]:
        return self.collector_service.start_capture(
            actor=actor,
            role=role,
            ip=ip,
            iface=payload.iface,
            bpf_filter=payload.bpf_filter,
            idle_timeout=payload.idle_timeout,
            min_packets=payload.min_packets,
            capture_label=payload.capture_label,
        )

    def stop_capture(self, actor: str, role: str, ip: str) -> dict[str, object]:
        return self.collector_service.stop_capture(actor=actor, role=role, ip=ip)

    def _run_cleanup(self) -> None:
        db_result = self.database_service.cleanup_expired_data()
        if any(db_result.values()):
            self.audit_service.append(
                level="info",
                action="retention_cleanup",
                message=(
                    f"Cleanup completed: capture_sessions={db_result['deleted_capture_sessions']}, "
                    f"predictions={db_result['deleted_predictions']}, logs={db_result['deleted_logs']}, "
                    f"files={db_result['deleted_files']}"
                ),
                actor="system",
                role="service",
                ip="127.0.0.1",
            )