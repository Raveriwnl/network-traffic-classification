from __future__ import annotations

import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.app.services.database_service import DatabaseService


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def to_iso(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


def parse_optional_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    candidate = value.replace("Z", "+00:00")
    return datetime.fromisoformat(candidate)


@dataclass(slots=True)
class AuditLogEntry:
    level: str
    action: str
    message: str
    actor: str | None = None
    role: str | None = None
    ip: str | None = None
    timestamp: datetime = field(default_factory=utc_now)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> dict[str, str | None]:
        return {
            "id": self.id,
            "timestamp": to_iso(self.timestamp),
            "level": self.level,
            "action": self.action,
            "message": self.message,
            "actor": self.actor,
            "role": self.role,
            "ip": self.ip,
        }


class AuditService:
    def __init__(self, max_entries: int = 500, database_service: DatabaseService | None = None) -> None:
        self._entries: deque[AuditLogEntry] = deque(maxlen=max_entries)
        self._database_service = database_service

    def append(
        self,
        *,
        level: str,
        action: str,
        message: str,
        actor: str | None = None,
        role: str | None = None,
        ip: str | None = None,
    ) -> None:
        entry = AuditLogEntry(level=level, action=action, message=message, actor=actor, role=role, ip=ip)
        self._entries.appendleft(entry)
        if self._database_service is not None:
            self._database_service.insert_audit_log(entry.to_dict())

    def query(
        self,
        *,
        limit: int,
        level: str | None = None,
        actor: str | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
    ) -> list[dict[str, str | None]]:
        if self._database_service is not None:
            return self._database_service.query_audit_logs(
                limit=limit,
                level=level,
                actor=actor,
                start_time=start_time,
                end_time=end_time,
            )
        start_dt = parse_optional_datetime(start_time)
        end_dt = parse_optional_datetime(end_time)
        items: list[dict[str, str | None]] = []
        for entry in self._entries:
            if level and entry.level != level:
                continue
            if actor and entry.actor != actor:
                continue
            if start_dt and entry.timestamp < start_dt:
                continue
            if end_dt and entry.timestamp > end_dt:
                continue
            items.append(entry.to_dict())
            if len(items) >= limit:
                break
        return items