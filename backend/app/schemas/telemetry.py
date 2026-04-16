from __future__ import annotations

from pydantic import BaseModel


class StreamSnapshot(BaseModel):
    flow_id: int
    packet_size: int
    iat: float
    direction: str
    duration_ms: int
    packet_count: int


class MetricsSnapshot(BaseModel):
    accuracy: float
    recall: float
    inference_latency_ms: float
    power_w: float
    flows_per_sec: float


class PredictionSnapshot(BaseModel):
    prediction_id: int
    class_id: int
    class_name: str
    confidence: float
    model_name: str
    model_version: str


class TelemetryResponse(BaseModel):
    timestamp: str
    stream: StreamSnapshot
    metrics: MetricsSnapshot
    prediction: PredictionSnapshot
    distribution: dict[str, float]