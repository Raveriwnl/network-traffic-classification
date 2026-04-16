from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class LogEntryResponse(BaseModel):
    id: str
    timestamp: str
    level: Literal["info", "warn", "error"]
    action: str
    message: str
    actor: str | None = None
    role: str | None = None
    ip: str | None = None


class LogsResponse(BaseModel):
    logs: list[LogEntryResponse]


class CaptureStartRequest(BaseModel):
    iface: str = Field(min_length=1, max_length=128)
    bpf_filter: str = Field(default="tcp or udp", max_length=255)
    idle_timeout: int = Field(default=5, ge=1, le=300)
    min_packets: int = Field(default=6, ge=1, le=10_000)
    capture_label: str = Field(default="online_session", min_length=1, max_length=128)


class CaptureSessionResponse(BaseModel):
    id: int
    session_name: str
    iface: str
    bpf_filter: str
    idle_timeout_sec: float
    min_packets: int
    status: str
    started_by: str
    started_at: str
    stopped_at: str | None = None
    output_dir: str | None = None
    packet_csv: str | None = None
    flow_metadata_csv: str | None = None
    summary_json: str | None = None
    resolved_ifaces: list[str] = Field(default_factory=list)
    total_flows_captured: int | None = None
    total_flows_saved: int | None = None
    total_packets_saved: int | None = None
    candidate_flows: int | None = None
    stop_reason: str | None = None
    error_message: str | None = None