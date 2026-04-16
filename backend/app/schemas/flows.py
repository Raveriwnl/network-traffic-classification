from __future__ import annotations

from pydantic import BaseModel


class FlowItemResponse(BaseModel):
    flow_id: int
    session_id: int
    protocol: str
    src_ip: str
    src_port: int
    dst_ip: str
    dst_port: int
    first_seen_at: str
    last_seen_at: str
    duration_ms: int
    packet_count: int
    eligible_for_inference: bool
    latest_prediction: str | None = None


class FlowListResponse(BaseModel):
    items: list[FlowItemResponse]
    total: int
    page: int
    page_size: int


class PredictionRecordResponse(BaseModel):
    id: int
    flow_id: int
    predicted_class: str
    confidence: float
    status: str
    inference_latency_ms: float
    device: str
    predicted_at: str
    distribution: dict[str, float]


class PredictionsLatestResponse(BaseModel):
    items: list[PredictionRecordResponse]


class FlowDetailResponse(BaseModel):
    flow: FlowItemResponse
    metadata: dict[str, object]
    feature_summary: dict[str, object]
    packet_sequence_summary: list[dict[str, object]]
    predictions: list[PredictionRecordResponse]