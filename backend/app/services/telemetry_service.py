from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone

from backend.app.core.config import Settings
from backend.app.services.audit_service import to_iso
from backend.app.services.database_service import DatabaseService
from backend.app.services.elastic_log_service import ElasticModelLogService
from backend.app.services.flow_service import FlowService


@dataclass(slots=True)
class PredictionMetric:
    actual: str | None
    predicted: str
    latency_ms: float
    timestamp: datetime


class TelemetryService:
    def __init__(
        self,
        settings: Settings,
        flow_service: FlowService,
        database_service: DatabaseService,
        elastic_log_service: ElasticModelLogService,
    ) -> None:
        self.settings = settings
        self.flow_service = flow_service
        self.database_service = database_service
        self.elastic_log_service = elastic_log_service
        self._latest_snapshot: dict[str, object] | None = None
        self._metrics_window: deque[PredictionMetric] = deque(maxlen=256)
        self._last_prediction_id: int | None = None

    def _now_iso(self) -> str:
        return to_iso(datetime.now(timezone.utc))

    def _empty_snapshot(self) -> dict[str, object]:
        return {
            "timestamp": self._now_iso(),
            "stream": {
                "flow_id": 0,
                "packet_size": 0,
                "iat": 0.0,
                "direction": "downlink",
                "duration_ms": 0,
                "packet_count": 0,
            },
            "metrics": {
                "accuracy": self._compute_accuracy(),
                "recall": self._compute_recall(),
                "inference_latency_ms": 0.0,
                "power_w": self._estimate_power(0.0),
                "flows_per_sec": self._compute_flows_per_sec(),
            },
            "prediction": {
                "prediction_id": 0,
                "class_id": -1,
                "class_name": "waiting",
                "confidence": 0.0,
                "model_name": self.settings.model_name,
                "model_version": self.settings.model_version,
            },
            "distribution": self.flow_service.empty_distribution(),
        }

    def _build_stream_summary(self, latest_record: dict[str, object]) -> dict[str, object]:
        return {
            "flow_id": int(latest_record["flow_id"]),
            "packet_size": int(latest_record.get("latest_packet_size") or 0),
            "iat": round(float(latest_record.get("latest_iat_ms") or 0.0), 3),
            "direction": "uplink" if int(latest_record.get("latest_direction") or 0) == 1 else "downlink",
            "duration_ms": int(latest_record.get("duration_ms") or 0),
            "packet_count": int(latest_record.get("packet_count") or 0),
        }

    def _compute_accuracy(self) -> float:
        labeled_items = [item for item in self._metrics_window if item.actual]
        if not labeled_items:
            return 0.0
        hits = sum(1 for item in labeled_items if item.actual == item.predicted)
        return round(hits / len(labeled_items), 4)

    def _compute_recall(self) -> float:
        labeled_items = [item for item in self._metrics_window if item.actual]
        if not labeled_items:
            return 0.0
        recalls: list[float] = []
        for class_name in self.flow_service.class_names:
            actual_items = [item for item in labeled_items if item.actual == class_name]
            if not actual_items:
                continue
            hits = sum(1 for item in actual_items if item.predicted == class_name)
            recalls.append(hits / len(actual_items))
        if not recalls:
            return 0.0
        return round(sum(recalls) / len(recalls), 4)

    def _compute_flows_per_sec(self) -> float:
        if len(self._metrics_window) < 2:
            return 0.0
        elapsed = (self._metrics_window[-1].timestamp - self._metrics_window[0].timestamp).total_seconds()
        if elapsed <= 0:
            return float(len(self._metrics_window))
        return round(len(self._metrics_window) / elapsed, 2)

    def _estimate_power(self, latency_ms: float) -> float:
        return round(3.8 + min(latency_ms, 15.0) * 0.11, 3)

    def generate_next_snapshot(self) -> dict[str, object]:
        latest_record = self.database_service.get_latest_captured_prediction_record()
        if latest_record is None:
            self._latest_snapshot = self._empty_snapshot()
            return self._latest_snapshot

        prediction_id = int(latest_record["prediction_id"])
        timestamp = datetime.now(timezone.utc)
        if prediction_id != self._last_prediction_id:
            self._metrics_window.append(
                PredictionMetric(
                    actual=latest_record.get("actual_label"),
                    predicted=str(latest_record["predicted_class"]),
                    latency_ms=float(latest_record["inference_latency_ms"]),
                    timestamp=_parse_prediction_time(str(latest_record["predicted_at"])),
                )
            )
            self._last_prediction_id = prediction_id

        distribution = latest_record.get("distribution") or self.flow_service.empty_distribution()
        snapshot = {
            "timestamp": str(latest_record["predicted_at"]),
            "stream": self._build_stream_summary(latest_record),
            "metrics": {
                "accuracy": self._compute_accuracy(),
                "recall": self._compute_recall(),
                "inference_latency_ms": float(latest_record["inference_latency_ms"]),
                "power_w": self._estimate_power(float(latest_record["inference_latency_ms"])),
                "flows_per_sec": self._compute_flows_per_sec(),
            },
            "prediction": {
                "prediction_id": prediction_id,
                "class_id": int(latest_record.get("class_id", -1)),
                "class_name": str(latest_record["predicted_class"]),
                "confidence": float(latest_record["confidence"]),
                "model_name": self.settings.model_name,
                "model_version": self.settings.model_version,
            },
            "distribution": dict(distribution),
        }
        self._latest_snapshot = snapshot
        return snapshot

    def get_latest_snapshot(self) -> dict[str, object]:
        if self._latest_snapshot is None:
            return self.generate_next_snapshot()
        return self._latest_snapshot

    def list_recent_predictions(self, limit: int) -> dict[str, object]:
        return {"items": self.database_service.list_recent_predictions(limit, origin="captured")}


def _parse_prediction_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))