from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from typing import Any

from data_collection.traffic_collector import CaptureConfig, collect_traffic
from preprocess.huawei_bin_preprocess import build_flow_tensor

from backend.app.core.config import Settings
from backend.app.services.audit_service import AuditService, to_iso, utc_now
from backend.app.services.database_service import DatabaseService
from backend.app.services.model_service import ModelService


CATURED_FLOW_ID_BASE = 2_000_000_000


@dataclass(slots=True)
class CaptureSessionState:
    id: int
    session_name: str
    iface: str
    bpf_filter: str
    idle_timeout_sec: float
    min_packets: int
    status: str
    started_by: str
    started_at: str
    output_dir: str
    stopped_at: str | None = None
    packet_csv: str | None = None
    flow_metadata_csv: str | None = None
    summary_json: str | None = None
    resolved_ifaces: list[str] = field(default_factory=list)
    total_flows_captured: int | None = None
    total_flows_saved: int | None = None
    total_packets_saved: int | None = None
    candidate_flows: int | None = None
    stop_reason: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "session_name": self.session_name,
            "iface": self.iface,
            "bpf_filter": self.bpf_filter,
            "idle_timeout_sec": self.idle_timeout_sec,
            "min_packets": self.min_packets,
            "status": self.status,
            "started_by": self.started_by,
            "started_at": self.started_at,
            "stopped_at": self.stopped_at,
            "output_dir": self.output_dir,
            "packet_csv": self.packet_csv,
            "flow_metadata_csv": self.flow_metadata_csv,
            "summary_json": self.summary_json,
            "resolved_ifaces": list(self.resolved_ifaces),
            "total_flows_captured": self.total_flows_captured,
            "total_flows_saved": self.total_flows_saved,
            "total_packets_saved": self.total_packets_saved,
            "candidate_flows": self.candidate_flows,
            "stop_reason": self.stop_reason,
            "error_message": self.error_message,
        }


class CollectorService:
    def __init__(
        self,
        settings: Settings,
        audit_service: AuditService,
        database_service: DatabaseService,
        model_service: ModelService,
    ) -> None:
        self.settings = settings
        self.audit_service = audit_service
        self.database_service = database_service
        self.model_service = model_service
        self._lock = threading.RLock()
        self._capture_next_id = 1
        self._capture_session: CaptureSessionState | None = None
        self._capture_thread: threading.Thread | None = None
        self._stop_event: threading.Event | None = None

    def start_capture(
        self,
        *,
        actor: str,
        role: str,
        ip: str,
        iface: str,
        bpf_filter: str,
        idle_timeout: int,
        min_packets: int,
        capture_label: str,
    ) -> dict[str, object]:
        with self._lock:
            if self._capture_session and self._capture_session.status == "running":
                raise ValueError("A capture session is already running.")

            output_dir = self.settings.capture_output_dir.resolve()
            output_dir.mkdir(parents=True, exist_ok=True)
            session = CaptureSessionState(
                id=self._capture_next_id,
                session_name=capture_label,
                iface=iface,
                bpf_filter=bpf_filter,
                idle_timeout_sec=float(idle_timeout),
                min_packets=min_packets,
                status="running",
                started_by=actor,
                started_at=to_iso(utc_now()),
                output_dir=str(output_dir),
            )
            self._capture_next_id += 1
            self._capture_session = session
            self._stop_event = threading.Event()
            thread = threading.Thread(
                target=self._run_capture,
                args=(session.id, actor, role, ip),
                name=f"collector-session-{session.id}",
                daemon=True,
            )
            self._capture_thread = thread
            thread.start()
            self.database_service.upsert_capture_session(session.to_dict())

        self.audit_service.append(
            level="info",
            action="capture_start",
            message=f"Capture session started on iface={iface} with label={capture_label}.",
            actor=actor,
            role=role,
            ip=ip,
        )
        return session.to_dict()

    def stop_capture(self, *, actor: str, role: str, ip: str) -> dict[str, object]:
        with self._lock:
            session = self._capture_session
            thread = self._capture_thread
            stop_event = self._stop_event
            if not session or session.status != "running" or thread is None or stop_event is None:
                raise ValueError("No capture session is currently running.")
            stop_event.set()

        self.audit_service.append(
            level="info",
            action="capture_stop_requested",
            message=f"Capture stop requested for session {session.id}.",
            actor=actor,
            role=role,
            ip=ip,
        )

        thread.join(timeout=self.settings.capture_stop_timeout_sec)
        if thread.is_alive():
            raise RuntimeError("Capture worker did not stop within the configured timeout.")

        with self._lock:
            assert self._capture_session is not None
            self.database_service.upsert_capture_session(self._capture_session.to_dict())
            return self._capture_session.to_dict()

    def shutdown(self) -> None:
        with self._lock:
            thread = self._capture_thread
            stop_event = self._stop_event
            session = self._capture_session
            if thread is None or stop_event is None or session is None or session.status != "running":
                return
            stop_event.set()
        thread.join(timeout=self.settings.capture_stop_timeout_sec)

    def _run_capture(self, session_id: int, actor: str, role: str, ip: str) -> None:
        with self._lock:
            assert self._capture_session is not None
            session = self._capture_session
            assert self._stop_event is not None
            stop_event = self._stop_event
            config = CaptureConfig(
                iface=session.iface,
                duration=self.settings.capture_duration_sec,
                flush_interval_sec=self.settings.capture_flush_interval_sec,
                idle_timeout=session.idle_timeout_sec,
                min_packets=session.min_packets,
                output_dir=self.settings.capture_output_dir.resolve(),
                capture_label=session.session_name,
                bpf_filter=session.bpf_filter,
                target_classes=(),
            )

        try:
            result = collect_traffic(
                config,
                stop_event=stop_event,
                logger=self._collector_logger(session_id),
                batch_callback=self._capture_batch_writer(session_id),
            )
        except Exception as exc:
            with self._lock:
                if self._capture_session is not None and self._capture_session.id == session_id:
                    self._capture_session.status = "failed"
                    self._capture_session.stopped_at = to_iso(utc_now())
                    self._capture_session.error_message = str(exc)
                    self._capture_thread = None
                    self._stop_event = None
                    self.database_service.upsert_capture_session(self._capture_session.to_dict())
            self.audit_service.append(
                level="error",
                action="capture_failed",
                message=f"Capture session {session_id} failed: {exc}",
                actor=actor,
                role=role,
                ip=ip,
            )
            return

        self._persist_and_predict_capture(session_id, result)

        with self._lock:
            if self._capture_session is not None and self._capture_session.id == session_id:
                self._capture_session.status = "stopped"
                self._capture_session.stopped_at = result.get("capture_stopped_at")
                self._capture_session.packet_csv = result.get("packet_csv")
                self._capture_session.flow_metadata_csv = result.get("flow_metadata_csv")
                self._capture_session.summary_json = result.get("summary_json")
                self._capture_session.resolved_ifaces = list(result.get("resolved_ifaces", []))
                self._capture_session.total_flows_captured = result.get("total_flows_captured")
                self._capture_session.total_flows_saved = result.get("total_flows_saved")
                self._capture_session.total_packets_saved = result.get("total_packets_saved")
                self._capture_session.candidate_flows = result.get("candidate_flows")
                self._capture_session.stop_reason = result.get("stop_reason")
                self._capture_thread = None
                self._stop_event = None
                self.database_service.upsert_capture_session(self._capture_session.to_dict())

        self.audit_service.append(
            level="info",
            action="capture_stopped",
            message=(
                f"Capture session {session_id} finished with stop_reason={result.get('stop_reason')} "
                f"and saved {result.get('total_packets_saved', 0)} packets."
            ),
            actor=actor,
            role=role,
            ip=ip,
        )

    def _collector_logger(self, session_id: int):
        def _log(message: str) -> None:
            self.audit_service.append(
                level="info",
                action="capture_runtime",
                message=f"session={session_id} {message}",
                actor="collector",
                role="service",
                ip="127.0.0.1",
            )

        return _log

    def _capture_batch_writer(self, session_id: int):
        def _write(batch: dict[str, Any]) -> None:
            self._persist_and_predict_capture(session_id, batch, update_session_progress=True)

        return _write

    def _persist_and_predict_capture(
        self,
        session_id: int,
        result: dict[str, Any],
        *,
        update_session_progress: bool = False,
    ) -> None:
        flow_rows = list(result.get("flow_rows_data", []))
        packet_rows = list(result.get("packet_rows_data", []))
        if not flow_rows:
            return

        packet_rows_by_flow: dict[int, list[dict[str, Any]]] = {}
        for packet_row in packet_rows:
            packet_rows_by_flow.setdefault(int(packet_row["flow_id"]), []).append(packet_row)

        now_iso = to_iso(utc_now())
        db_flows: list[dict[str, Any]] = []
        db_packets: list[dict[str, Any]] = []
        db_predictions: list[dict[str, Any]] = []

        for flow_row in flow_rows:
            source_flow_id = int(flow_row["flow_id"])
            global_flow_id = CATURED_FLOW_ID_BASE + (session_id * 1_000_000) + source_flow_id
            packet_group = sorted(packet_rows_by_flow.get(source_flow_id, []), key=lambda item: int(item["arrive_time"]))
            if not packet_group:
                continue

            tensor_input = build_flow_tensor(
                self._packet_rows_to_frame(source_flow_id, packet_group),
                bins=self.settings.bins,
                window_ms=float(self.settings.window_ms),
            )
            inference = self.model_service.infer(tensor_input)
            prediction_payload = {
                "id": global_flow_id,
                "flow_id": global_flow_id,
                "predicted_class": str(inference["class_name"]),
                "confidence": float(inference["confidence"]),
                "status": "succeeded",
                "inference_latency_ms": float(inference["inference_latency_ms"]),
                "device": str(inference["device"]),
                "predicted_at": now_iso,
                "distribution": dict(inference["distribution"]),
                "actual_label": None,
            }
            metadata = {
                "origin": "captured",
                "source_flow_id": source_flow_id,
                "candidate_labels": str(flow_row.get("candidate_labels") or ""),
                "is_target_candidate": int(flow_row.get("is_target_candidate") or 0),
                "tls_sni": str(flow_row.get("tls_sni") or ""),
                "http_host": str(flow_row.get("http_host") or ""),
                "dns_queries": str(flow_row.get("dns_queries") or ""),
                "capture_summary_json": str(result.get("summary_json") or ""),
            }
            feature_summary = {
                "window_ms": self.settings.window_ms,
                "bins": self.settings.bins,
                "feature_count": self.settings.feature_count,
                "packet_count": len(packet_group),
                "non_empty_bins": int((tensor_input[:, 0] > 0).sum()),
                "mean_packet_size": round(float(tensor_input[:, 1][tensor_input[:, 0] > 0].mean()) if (tensor_input[:, 0] > 0).any() else 0.0, 3),
            }

            db_flows.append(
                {
                    "flow_id": global_flow_id,
                    "source_flow_id": source_flow_id,
                    "origin": "captured",
                    "sample_index": None,
                    "session_id": session_id,
                    "protocol": str(flow_row["proto"]),
                    "src_ip": str(flow_row["src_ip"]),
                    "src_port": int(flow_row["src_port"]),
                    "dst_ip": str(flow_row["dst_ip"]),
                    "dst_port": int(flow_row["dst_port"]),
                    "first_seen_at": str(flow_row["first_seen_iso"]),
                    "last_seen_at": str(flow_row["last_seen_iso"]),
                    "duration_ms": int(flow_row["duration_ms"]),
                    "packet_count": int(flow_row["packet_count"]),
                    "eligible_for_inference": 1,
                    "latest_prediction": str(inference["class_name"]),
                    "flow_status": "predicted",
                    "label": None,
                    "source_file": str(result.get("flow_metadata_csv") or ""),
                    "feature_summary_json": json.dumps(feature_summary, ensure_ascii=False),
                    "metadata_json": json.dumps(metadata, ensure_ascii=False),
                    "created_at": now_iso,
                    "updated_at": now_iso,
                }
            )
            for packet_row in packet_group:
                db_packets.append(
                    {
                        "flow_id": global_flow_id,
                        "arrive_time_ms": int(packet_row["arrive_time"]),
                        "direction": int(packet_row["direction"]),
                        "pkt_len": int(packet_row["pkt_len"]),
                        "created_at": now_iso,
                    }
                )
            db_predictions.append(prediction_payload)
        self.database_service.upsert_flows(db_flows)
        self.database_service.replace_captured_packets(db_packets)
        for prediction in db_predictions:
            self.database_service.insert_prediction(prediction)

        if update_session_progress:
            self._update_capture_session_progress(session_id, result)

    def _update_capture_session_progress(self, session_id: int, result: dict[str, Any]) -> None:
        with self._lock:
            if self._capture_session is None or self._capture_session.id != session_id:
                return
            self._capture_session.total_flows_captured = result.get("total_flows_captured")
            self._capture_session.total_flows_saved = result.get("total_flows_saved")
            self._capture_session.total_packets_saved = result.get("total_packets_saved")
            self._capture_session.candidate_flows = result.get("candidate_flows")
            self.database_service.upsert_capture_session(self._capture_session.to_dict())

    @staticmethod
    def _packet_rows_to_frame(source_flow_id: int, packet_rows: list[dict[str, Any]]):
        import pandas as pd

        return pd.DataFrame(
            {
                "flow_id": [source_flow_id for _ in packet_rows],
                "arrive_time": [float(item["arrive_time"]) for item in packet_rows],
                "direction": [1 - int(item["direction"]) for item in packet_rows],
                "pkt_len": [int(item["pkt_len"]) for item in packet_rows],
            }
        )