from __future__ import annotations

import json
from typing import Any

from backend.app.core.config import Settings
from backend.app.core.errors import AppError
from backend.app.services.database_service import DatabaseService


class FlowService:
    def __init__(self, settings: Settings, database_service: DatabaseService) -> None:
        self.settings = settings
        self.database_service = database_service
        with settings.huawei_schema_path.open("r", encoding="utf-8") as file_obj:
            schema = json.load(file_obj)
        self.class_names = [str(item) for item in schema["classes"]]

    def empty_distribution(self) -> dict[str, float]:
        return {class_name: 0.0 for class_name in self.class_names}

    def get_prediction_history(self, flow_id: int) -> list[dict[str, object]]:
        return self.database_service.get_predictions_for_flow(flow_id)

    def get_flow_detail(self, flow_id: int) -> dict[str, object]:
        db_row = self.database_service.get_flow_row(flow_id)
        if db_row is None:
            raise AppError(404, "Flow not found.", "FLOW_NOT_FOUND")
        if db_row.get("origin") != "captured":
            raise AppError(404, "Flow not found.", "FLOW_NOT_FOUND")

        metadata = json.loads(db_row.get("metadata_json") or "{}")
        feature_summary = json.loads(db_row.get("feature_summary_json") or "{}")
        packet_rows = self.database_service.get_flow_packet_summary(flow_id)
        packet_summary = [
            {
                "arrive_time_ms": row["arrive_time_ms"],
                "direction": "uplink" if row["direction"] == 0 else "downlink",
                "pkt_len": row["pkt_len"],
            }
            for row in packet_rows
        ]
        return {
            "flow": {
                "flow_id": int(db_row["flow_id"]),
                "session_id": int(db_row["session_id"]),
                "protocol": str(db_row["protocol"]),
                "src_ip": str(db_row["src_ip"]),
                "src_port": int(db_row["src_port"]),
                "dst_ip": str(db_row["dst_ip"]),
                "dst_port": int(db_row["dst_port"]),
                "first_seen_at": str(db_row["first_seen_at"]),
                "last_seen_at": str(db_row["last_seen_at"]),
                "duration_ms": int(db_row["duration_ms"]),
                "packet_count": int(db_row["packet_count"]),
                "eligible_for_inference": bool(db_row["eligible_for_inference"]),
                "latest_prediction": db_row["latest_prediction"],
            },
            "metadata": metadata,
            "feature_summary": feature_summary,
            "packet_sequence_summary": packet_summary,
            "predictions": self.get_prediction_history(flow_id),
        }

    def list_flows(
        self,
        *,
        page: int,
        page_size: int,
        source_ip: str | None,
        dest_ip: str | None,
        protocol: str | None,
        eligible_for_inference: bool | None,
        predicted_class: str | None,
        start_time: str | None,
        end_time: str | None,
    ) -> dict[str, object]:
        return self.database_service.list_flows(
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