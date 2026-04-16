from __future__ import annotations

from datetime import timedelta
from typing import Any

from elasticsearch import Elasticsearch

from backend.app.core.config import Settings
from backend.app.services.audit_service import to_iso, utc_now


class ElasticModelLogService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._enabled = bool(settings.elasticsearch_url)
        self._index_ready = False
        self._client: Elasticsearch | None = None
        if self._enabled:
            kwargs: dict[str, Any] = {"hosts": [settings.elasticsearch_url]}
            if settings.elasticsearch_username and settings.elasticsearch_password:
                kwargs["basic_auth"] = (settings.elasticsearch_username, settings.elasticsearch_password)
            self._client = Elasticsearch(**kwargs)

    def close(self) -> None:
        if self._client is not None:
            self._client.close()

    def _ensure_index(self) -> None:
        if not self._enabled or self._client is None or self._index_ready:
            return
        if not self._client.indices.exists(index=self.settings.elasticsearch_index):
            self._client.indices.create(
                index=self.settings.elasticsearch_index,
                mappings={
                    "properties": {
                        "timestamp": {"type": "date"},
                        "source": {"type": "keyword"},
                        "flow_id": {"type": "long"},
                        "prediction_id": {"type": "long"},
                        "predicted_class": {"type": "keyword"},
                        "class_id": {"type": "integer"},
                        "confidence": {"type": "float"},
                        "inference_latency_ms": {"type": "float"},
                        "device": {"type": "keyword"},
                        "actual_label": {"type": "keyword"},
                        "packet_count": {"type": "integer"},
                        "duration_ms": {"type": "integer"},
                        "status": {"type": "keyword"},
                        "distribution": {"type": "object", "enabled": True},
                        "metadata": {"type": "object", "enabled": True},
                    }
                },
            )
        self._index_ready = True

    def log_inference_event(
        self,
        *,
        source: str,
        flow_id: int,
        prediction_id: int,
        predicted_class: str,
        class_id: int,
        confidence: float,
        distribution: dict[str, float],
        inference_latency_ms: float,
        device: str,
        actual_label: str | None,
        packet_count: int,
        duration_ms: int,
        status: str,
        timestamp: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not self._enabled or self._client is None:
            return
        document = {
            "timestamp": timestamp or to_iso(utc_now()),
            "source": source,
            "flow_id": flow_id,
            "prediction_id": prediction_id,
            "predicted_class": predicted_class,
            "class_id": class_id,
            "confidence": confidence,
            "distribution": distribution,
            "inference_latency_ms": inference_latency_ms,
            "device": device,
            "actual_label": actual_label,
            "packet_count": packet_count,
            "duration_ms": duration_ms,
            "status": status,
            "metadata": metadata or {},
        }
        try:
            self._ensure_index()
            self._client.index(index=self.settings.elasticsearch_index, document=document)
        except Exception:
            return

    def cleanup_expired_logs(self) -> int:
        if not self._enabled or self._client is None:
            return 0
        cutoff_iso = to_iso(utc_now() - timedelta(days=self.settings.data_retention_days))
        try:
            self._ensure_index()
            result = self._client.delete_by_query(
                index=self.settings.elasticsearch_index,
                query={"range": {"timestamp": {"lt": cutoff_iso}}},
                conflicts="proceed",
                refresh=True,
            )
        except Exception:
            return 0
        return int(result.get("deleted", 0))

    def purge_all_logs(self) -> int:
        if not self._enabled or self._client is None:
            return 0
        try:
            if not self._client.indices.exists(index=self.settings.elasticsearch_index):
                return 0
            result = self._client.delete_by_query(
                index=self.settings.elasticsearch_index,
                query={"match_all": {}},
                conflicts="proceed",
                refresh=True,
            )
        except Exception:
            return 0
        return int(result.get("deleted", 0))