from __future__ import annotations

import json
import threading
from collections.abc import Iterable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from psycopg import connect
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from backend.app.core.config import Settings
from backend.app.services.audit_service import parse_optional_datetime, to_iso, utc_now


def _parse_datetime(value: str | datetime | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    return parse_optional_datetime(value)


def _require_datetime(value: str | datetime, field_name: str) -> datetime:
    parsed = _parse_datetime(value)
    if parsed is None:
        raise ValueError(f"{field_name} is required.")
    return parsed


def _jsonb(value: Any) -> Jsonb:
    if isinstance(value, str):
        return Jsonb(json.loads(value))
    return Jsonb(value)


def _normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, datetime):
            data[key] = to_iso(value.astimezone(timezone.utc))
        else:
            data[key] = value
    return data


class DatabaseService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._lock = threading.RLock()
        self._connection = connect(settings.postgres_dsn, autocommit=True, row_factory=dict_row)
        self._init_schema()

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def _init_schema(self) -> None:
        statements = [
            """
            CREATE TABLE IF NOT EXISTS users (
                id BIGSERIAL PRIMARY KEY,
                username VARCHAR(64) NOT NULL UNIQUE,
                password_hash VARCHAR(255) NOT NULL,
                display_name VARCHAR(128) NOT NULL,
                role VARCHAR(16) NOT NULL,
                status VARCHAR(16) NOT NULL DEFAULT 'active',
                last_login_at TIMESTAMPTZ,
                created_at TIMESTAMPTZ NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS audit_logs (
                id TEXT PRIMARY KEY,
                timestamp TIMESTAMPTZ NOT NULL,
                level VARCHAR(16) NOT NULL,
                action VARCHAR(64) NOT NULL,
                message TEXT NOT NULL,
                actor VARCHAR(64),
                role VARCHAR(16),
                ip VARCHAR(64)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS capture_sessions (
                id BIGINT PRIMARY KEY,
                session_name VARCHAR(128) NOT NULL,
                iface VARCHAR(128) NOT NULL,
                bpf_filter VARCHAR(255) NOT NULL,
                idle_timeout_sec NUMERIC(8,3) NOT NULL,
                min_packets INTEGER NOT NULL,
                status VARCHAR(16) NOT NULL,
                started_by VARCHAR(64) NOT NULL,
                started_at TIMESTAMPTZ NOT NULL,
                stopped_at TIMESTAMPTZ,
                output_dir TEXT,
                packet_csv TEXT,
                flow_metadata_csv TEXT,
                summary_json TEXT,
                resolved_ifaces JSONB NOT NULL DEFAULT '[]'::jsonb,
                total_flows_captured INTEGER,
                total_flows_saved INTEGER,
                total_packets_saved INTEGER,
                candidate_flows INTEGER,
                stop_reason VARCHAR(64),
                error_message TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS flows (
                flow_id BIGINT PRIMARY KEY,
                source_flow_id BIGINT,
                origin VARCHAR(32) NOT NULL,
                sample_index INTEGER,
                session_id BIGINT NOT NULL,
                protocol VARCHAR(8) NOT NULL,
                src_ip VARCHAR(64) NOT NULL,
                src_port INTEGER NOT NULL,
                dst_ip VARCHAR(64) NOT NULL,
                dst_port INTEGER NOT NULL,
                first_seen_at TIMESTAMPTZ NOT NULL,
                last_seen_at TIMESTAMPTZ NOT NULL,
                duration_ms INTEGER NOT NULL,
                packet_count INTEGER NOT NULL,
                eligible_for_inference BOOLEAN NOT NULL,
                latest_prediction VARCHAR(64),
                flow_status VARCHAR(16) NOT NULL,
                label VARCHAR(64),
                source_file TEXT,
                feature_summary_json JSONB,
                metadata_json JSONB,
                created_at TIMESTAMPTZ NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS flow_packets (
                id BIGSERIAL PRIMARY KEY,
                flow_id BIGINT NOT NULL REFERENCES flows(flow_id) ON DELETE CASCADE,
                arrive_time_ms INTEGER NOT NULL,
                direction SMALLINT NOT NULL,
                pkt_len INTEGER NOT NULL,
                created_at TIMESTAMPTZ NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id BIGINT PRIMARY KEY,
                flow_id BIGINT NOT NULL REFERENCES flows(flow_id) ON DELETE CASCADE,
                predicted_class VARCHAR(64) NOT NULL,
                confidence NUMERIC(8,6) NOT NULL,
                status VARCHAR(16) NOT NULL,
                inference_latency_ms NUMERIC(10,3) NOT NULL,
                device VARCHAR(32) NOT NULL,
                predicted_at TIMESTAMPTZ NOT NULL,
                distribution_json JSONB NOT NULL,
                actual_label VARCHAR(64)
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)",
            "CREATE INDEX IF NOT EXISTS idx_audit_logs_timestamp ON audit_logs(timestamp DESC)",
            "CREATE INDEX IF NOT EXISTS idx_capture_sessions_started_at ON capture_sessions(started_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_flows_origin_first_seen ON flows(origin, first_seen_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_flow_packets_flow_arrive_desc ON flow_packets(flow_id, arrive_time_ms DESC)",
            "CREATE INDEX IF NOT EXISTS idx_predictions_flow_predicted_at ON predictions(flow_id, predicted_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_predictions_predicted_at_desc ON predictions(predicted_at DESC)",
        ]
        with self._lock, self._connection.cursor() as cursor:
            for statement in statements:
                cursor.execute(statement)

    def ensure_default_users(self, users: Iterable[dict[str, Any]]) -> None:
        rows = []
        for user in users:
            rows.append(
                (
                    user["username"],
                    user["password_hash"],
                    user["display_name"],
                    user["role"],
                    user["status"],
                    _parse_datetime(user.get("last_login_at")),
                    _require_datetime(user["created_at"], "created_at"),
                    _require_datetime(user["updated_at"], "updated_at"),
                )
            )
        if not rows:
            return
        with self._lock, self._connection.cursor() as cursor:
            cursor.executemany(
                """
                INSERT INTO users(
                    username, password_hash, display_name, role, status, last_login_at, created_at, updated_at
                ) VALUES(%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT(username) DO NOTHING
                """,
                rows,
            )

    def get_user_by_username(self, username: str) -> dict[str, Any] | None:
        with self._lock, self._connection.cursor() as cursor:
            cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
            row = cursor.fetchone()
        return None if row is None else _normalize_row(row)

    def get_user_by_id(self, user_id: int) -> dict[str, Any] | None:
        with self._lock, self._connection.cursor() as cursor:
            cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
            row = cursor.fetchone()
        return None if row is None else _normalize_row(row)

    def list_users(self) -> list[dict[str, Any]]:
        with self._lock, self._connection.cursor() as cursor:
            cursor.execute("SELECT * FROM users ORDER BY id ASC")
            rows = cursor.fetchall()
        return [_normalize_row(row) for row in rows]

    def create_user(self, user: dict[str, Any]) -> dict[str, Any]:
        with self._lock, self._connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO users(username, password_hash, display_name, role, status, created_at, updated_at)
                VALUES(%s, %s, %s, %s, %s, %s, %s)
                RETURNING *
                """,
                (
                    user["username"],
                    user["password_hash"],
                    user["display_name"],
                    user["role"],
                    user["status"],
                    _require_datetime(user["created_at"], "created_at"),
                    _require_datetime(user["updated_at"], "updated_at"),
                ),
            )
            row = cursor.fetchone()
        return _normalize_row(row)

    def update_user(
        self,
        user_id: int,
        *,
        display_name: str | None,
        role: str | None,
        status: str | None,
        updated_at: str | datetime,
    ) -> dict[str, Any] | None:
        assignments = ["updated_at = %s"]
        params: list[Any] = [_require_datetime(updated_at, "updated_at")]
        if display_name is not None:
            assignments.append("display_name = %s")
            params.append(display_name)
        if role is not None:
            assignments.append("role = %s")
            params.append(role)
        if status is not None:
            assignments.append("status = %s")
            params.append(status)
        params.append(user_id)
        query = f"UPDATE users SET {', '.join(assignments)} WHERE id = %s RETURNING *"
        with self._lock, self._connection.cursor() as cursor:
            cursor.execute(query, params)
            row = cursor.fetchone()
        return None if row is None else _normalize_row(row)

    def update_user_login(self, username: str, *, last_login_at: str | datetime, updated_at: str | datetime) -> None:
        with self._lock, self._connection.cursor() as cursor:
            cursor.execute(
                "UPDATE users SET last_login_at = %s, updated_at = %s WHERE username = %s",
                (
                    _require_datetime(last_login_at, "last_login_at"),
                    _require_datetime(updated_at, "updated_at"),
                    username,
                ),
            )

    def insert_audit_log(self, entry: dict[str, Any]) -> None:
        with self._lock, self._connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO audit_logs(id, timestamp, level, action, message, actor, role, ip)
                VALUES(%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT(id) DO UPDATE SET
                    timestamp = EXCLUDED.timestamp,
                    level = EXCLUDED.level,
                    action = EXCLUDED.action,
                    message = EXCLUDED.message,
                    actor = EXCLUDED.actor,
                    role = EXCLUDED.role,
                    ip = EXCLUDED.ip
                """,
                (
                    entry["id"],
                    _require_datetime(entry["timestamp"], "timestamp"),
                    entry["level"],
                    entry["action"],
                    entry["message"],
                    entry.get("actor"),
                    entry.get("role"),
                    entry.get("ip"),
                ),
            )

    def query_audit_logs(
        self,
        *,
        limit: int,
        level: str | None,
        actor: str | None,
        start_time: str | None,
        end_time: str | None,
    ) -> list[dict[str, Any]]:
        clauses = []
        params: list[Any] = []
        if level:
            clauses.append("level = %s")
            params.append(level)
        if actor:
            clauses.append("actor = %s")
            params.append(actor)
        if start_time:
            clauses.append("timestamp >= %s")
            params.append(_parse_datetime(start_time))
        if end_time:
            clauses.append("timestamp <= %s")
            params.append(_parse_datetime(end_time))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = (
            "SELECT id, timestamp, level, action, message, actor, role, ip "
            f"FROM audit_logs {where} ORDER BY timestamp DESC LIMIT %s"
        )
        params.append(limit)
        with self._lock, self._connection.cursor() as cursor:
            cursor.execute(query, params)
            rows = cursor.fetchall()
        return [_normalize_row(row) for row in rows]

    def upsert_capture_session(self, session: dict[str, Any]) -> None:
        with self._lock, self._connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO capture_sessions(
                    id, session_name, iface, bpf_filter, idle_timeout_sec, min_packets,
                    status, started_by, started_at, stopped_at, output_dir, packet_csv,
                    flow_metadata_csv, summary_json, resolved_ifaces, total_flows_captured,
                    total_flows_saved, total_packets_saved, candidate_flows, stop_reason, error_message
                ) VALUES(
                    %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s, %s
                )
                ON CONFLICT(id) DO UPDATE SET
                    session_name = EXCLUDED.session_name,
                    iface = EXCLUDED.iface,
                    bpf_filter = EXCLUDED.bpf_filter,
                    idle_timeout_sec = EXCLUDED.idle_timeout_sec,
                    min_packets = EXCLUDED.min_packets,
                    status = EXCLUDED.status,
                    started_by = EXCLUDED.started_by,
                    started_at = EXCLUDED.started_at,
                    stopped_at = EXCLUDED.stopped_at,
                    output_dir = EXCLUDED.output_dir,
                    packet_csv = EXCLUDED.packet_csv,
                    flow_metadata_csv = EXCLUDED.flow_metadata_csv,
                    summary_json = EXCLUDED.summary_json,
                    resolved_ifaces = EXCLUDED.resolved_ifaces,
                    total_flows_captured = EXCLUDED.total_flows_captured,
                    total_flows_saved = EXCLUDED.total_flows_saved,
                    total_packets_saved = EXCLUDED.total_packets_saved,
                    candidate_flows = EXCLUDED.candidate_flows,
                    stop_reason = EXCLUDED.stop_reason,
                    error_message = EXCLUDED.error_message
                """,
                (
                    session["id"],
                    session["session_name"],
                    session["iface"],
                    session["bpf_filter"],
                    session["idle_timeout_sec"],
                    session["min_packets"],
                    session["status"],
                    session["started_by"],
                    _require_datetime(session["started_at"], "started_at"),
                    _parse_datetime(session.get("stopped_at")),
                    session.get("output_dir"),
                    session.get("packet_csv"),
                    session.get("flow_metadata_csv"),
                    session.get("summary_json"),
                    _jsonb(session.get("resolved_ifaces", [])),
                    session.get("total_flows_captured"),
                    session.get("total_flows_saved"),
                    session.get("total_packets_saved"),
                    session.get("candidate_flows"),
                    session.get("stop_reason"),
                    session.get("error_message"),
                ),
            )

    def upsert_flows(self, flows: Iterable[dict[str, Any]]) -> None:
        rows = list(flows)
        if not rows:
            return
        params = [
            (
                row["flow_id"],
                row.get("source_flow_id"),
                row["origin"],
                row.get("sample_index"),
                row["session_id"],
                row["protocol"],
                row["src_ip"],
                row["src_port"],
                row["dst_ip"],
                row["dst_port"],
                _require_datetime(row["first_seen_at"], "first_seen_at"),
                _require_datetime(row["last_seen_at"], "last_seen_at"),
                row["duration_ms"],
                row["packet_count"],
                bool(row["eligible_for_inference"]),
                row.get("latest_prediction"),
                row["flow_status"],
                row.get("label"),
                row.get("source_file"),
                _jsonb(row.get("feature_summary_json") or {}),
                _jsonb(row.get("metadata_json") or {}),
                _require_datetime(row["created_at"], "created_at"),
                _require_datetime(row["updated_at"], "updated_at"),
            )
            for row in rows
        ]
        with self._lock, self._connection.cursor() as cursor:
            cursor.executemany(
                """
                INSERT INTO flows(
                    flow_id, source_flow_id, origin, sample_index, session_id, protocol,
                    src_ip, src_port, dst_ip, dst_port, first_seen_at, last_seen_at,
                    duration_ms, packet_count, eligible_for_inference, latest_prediction,
                    flow_status, label, source_file, feature_summary_json, metadata_json,
                    created_at, updated_at
                ) VALUES(
                    %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    %s, %s
                )
                ON CONFLICT(flow_id) DO UPDATE SET
                    source_flow_id = EXCLUDED.source_flow_id,
                    origin = EXCLUDED.origin,
                    sample_index = EXCLUDED.sample_index,
                    session_id = EXCLUDED.session_id,
                    protocol = EXCLUDED.protocol,
                    src_ip = EXCLUDED.src_ip,
                    src_port = EXCLUDED.src_port,
                    dst_ip = EXCLUDED.dst_ip,
                    dst_port = EXCLUDED.dst_port,
                    first_seen_at = EXCLUDED.first_seen_at,
                    last_seen_at = EXCLUDED.last_seen_at,
                    duration_ms = EXCLUDED.duration_ms,
                    packet_count = EXCLUDED.packet_count,
                    eligible_for_inference = EXCLUDED.eligible_for_inference,
                    latest_prediction = EXCLUDED.latest_prediction,
                    flow_status = EXCLUDED.flow_status,
                    label = EXCLUDED.label,
                    source_file = EXCLUDED.source_file,
                    feature_summary_json = EXCLUDED.feature_summary_json,
                    metadata_json = EXCLUDED.metadata_json,
                    updated_at = EXCLUDED.updated_at
                """,
                params,
            )

    def replace_captured_packets(self, flow_packets: Iterable[dict[str, Any]]) -> None:
        rows = list(flow_packets)
        if not rows:
            return
        flow_ids = sorted({int(row["flow_id"]) for row in rows})
        with self._lock, self._connection.cursor() as cursor:
            cursor.execute("DELETE FROM flow_packets WHERE flow_id = ANY(%s)", (flow_ids,))
            cursor.executemany(
                """
                INSERT INTO flow_packets(flow_id, arrive_time_ms, direction, pkt_len, created_at)
                VALUES(%s, %s, %s, %s, %s)
                """,
                [
                    (
                        row["flow_id"],
                        row["arrive_time_ms"],
                        row["direction"],
                        row["pkt_len"],
                        _require_datetime(row["created_at"], "created_at"),
                    )
                    for row in rows
                ],
            )

    def update_flow_prediction(self, flow_id: int, predicted_class: str, flow_status: str) -> None:
        with self._lock, self._connection.cursor() as cursor:
            cursor.execute(
                "UPDATE flows SET latest_prediction = %s, flow_status = %s, updated_at = %s WHERE flow_id = %s",
                (predicted_class, flow_status, utc_now(), flow_id),
            )

    def insert_prediction(self, prediction: dict[str, Any]) -> None:
        with self._lock, self._connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO predictions(
                    id, flow_id, predicted_class, confidence, status, inference_latency_ms,
                    device, predicted_at, distribution_json, actual_label
                ) VALUES(
                    %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s
                )
                ON CONFLICT(id) DO UPDATE SET
                    flow_id = EXCLUDED.flow_id,
                    predicted_class = EXCLUDED.predicted_class,
                    confidence = EXCLUDED.confidence,
                    status = EXCLUDED.status,
                    inference_latency_ms = EXCLUDED.inference_latency_ms,
                    device = EXCLUDED.device,
                    predicted_at = EXCLUDED.predicted_at,
                    distribution_json = EXCLUDED.distribution_json,
                    actual_label = EXCLUDED.actual_label
                """,
                (
                    prediction["id"],
                    prediction["flow_id"],
                    prediction["predicted_class"],
                    prediction["confidence"],
                    prediction["status"],
                    prediction["inference_latency_ms"],
                    prediction["device"],
                    _require_datetime(prediction["predicted_at"], "predicted_at"),
                    _jsonb(prediction["distribution"]),
                    prediction.get("actual_label"),
                ),
            )

    def list_recent_predictions(self, limit: int) -> list[dict[str, Any]]:
        return self._list_recent_predictions(limit=limit, origin=None)

    def _list_recent_predictions(self, *, limit: int, origin: str | None) -> list[dict[str, Any]]:
        params: list[Any] = []
        where = ""
        if origin is not None:
            where = "WHERE f.origin = %s"
            params.append(origin)
        with self._lock, self._connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT p.id, p.flow_id, p.predicted_class, p.confidence, p.status, p.inference_latency_ms, p.device, p.predicted_at, p.distribution_json
                FROM predictions p
                INNER JOIN flows f ON f.flow_id = p.flow_id
                """
                + where
                + " ORDER BY p.predicted_at DESC LIMIT %s",
                [*params, limit],
            )
            rows = cursor.fetchall()
        items: list[dict[str, Any]] = []
        for row in rows:
            data = _normalize_row(row)
            distribution = data.pop("distribution_json")
            data["distribution"] = distribution if isinstance(distribution, dict) else json.loads(distribution)
            items.append(data)
        return items

    def list_recent_predictions(self, limit: int, origin: str | None = None) -> list[dict[str, Any]]:
        return self._list_recent_predictions(limit=limit, origin=origin)

    def get_predictions_for_flow(self, flow_id: int) -> list[dict[str, Any]]:
        with self._lock, self._connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, flow_id, predicted_class, confidence, status, inference_latency_ms, device, predicted_at, distribution_json
                FROM predictions WHERE flow_id = %s ORDER BY predicted_at DESC
                """,
                (flow_id,),
            )
            rows = cursor.fetchall()
        items: list[dict[str, Any]] = []
        for row in rows:
            data = _normalize_row(row)
            distribution = data.pop("distribution_json")
            data["distribution"] = distribution if isinstance(distribution, dict) else json.loads(distribution)
            items.append(data)
        return items

    def get_latest_prediction_map(self) -> dict[int, str]:
        query = """
        SELECT p.flow_id, p.predicted_class
        FROM predictions p
        INNER JOIN (
            SELECT flow_id, MAX(predicted_at) AS max_predicted_at
            FROM predictions GROUP BY flow_id
        ) latest ON latest.flow_id = p.flow_id AND latest.max_predicted_at = p.predicted_at
        """
        with self._lock, self._connection.cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()
        return {int(row["flow_id"]): str(row["predicted_class"]) for row in rows}

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
    ) -> dict[str, Any]:
        clauses = ["origin = 'captured'"]
        params: list[Any] = []
        if source_ip:
            clauses.append("src_ip = %s")
            params.append(source_ip)
        if dest_ip:
            clauses.append("dst_ip = %s")
            params.append(dest_ip)
        if protocol:
            clauses.append("protocol = %s")
            params.append(protocol)
        if eligible_for_inference is not None:
            clauses.append("eligible_for_inference = %s")
            params.append(eligible_for_inference)
        if predicted_class:
            clauses.append("latest_prediction = %s")
            params.append(predicted_class)
        if start_time:
            clauses.append("first_seen_at >= %s")
            params.append(_parse_datetime(start_time))
        if end_time:
            clauses.append("last_seen_at <= %s")
            params.append(_parse_datetime(end_time))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        count_query = f"SELECT COUNT(*) AS total FROM flows {where}"
        data_query = (
            "SELECT flow_id, session_id, protocol, src_ip, src_port, dst_ip, dst_port, first_seen_at, "
            "last_seen_at, duration_ms, packet_count, eligible_for_inference, latest_prediction "
            f"FROM flows {where} ORDER BY first_seen_at DESC LIMIT %s OFFSET %s"
        )
        offset = (page - 1) * page_size
        with self._lock, self._connection.cursor() as cursor:
            cursor.execute(count_query, params)
            total = int(cursor.fetchone()["total"])
            cursor.execute(data_query, [*params, page_size, offset])
            rows = cursor.fetchall()
        return {
            "items": [
                {
                    "flow_id": int(row["flow_id"]),
                    "session_id": int(row["session_id"]),
                    "protocol": str(row["protocol"]),
                    "src_ip": str(row["src_ip"]),
                    "src_port": int(row["src_port"]),
                    "dst_ip": str(row["dst_ip"]),
                    "dst_port": int(row["dst_port"]),
                    "first_seen_at": to_iso(row["first_seen_at"].astimezone(timezone.utc)),
                    "last_seen_at": to_iso(row["last_seen_at"].astimezone(timezone.utc)),
                    "duration_ms": int(row["duration_ms"]),
                    "packet_count": int(row["packet_count"]),
                    "eligible_for_inference": bool(row["eligible_for_inference"]),
                    "latest_prediction": row["latest_prediction"],
                }
                for row in rows
            ],
            "total": total,
            "page": page,
            "page_size": page_size,
        }

    def get_flow_row(self, flow_id: int) -> dict[str, Any] | None:
        with self._lock, self._connection.cursor() as cursor:
            cursor.execute("SELECT * FROM flows WHERE flow_id = %s", (flow_id,))
            row = cursor.fetchone()
        if row is None:
            return None
        data = _normalize_row(row)
        for json_key in ("feature_summary_json", "metadata_json"):
            value = data.get(json_key)
            if value is not None and not isinstance(value, str):
                data[json_key] = json.dumps(value, ensure_ascii=False)
        return data

    def get_flow_packet_summary(self, flow_id: int, limit: int = 64) -> list[dict[str, Any]]:
        with self._lock, self._connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT arrive_time_ms, direction, pkt_len
                FROM flow_packets WHERE flow_id = %s ORDER BY arrive_time_ms ASC LIMIT %s
                """,
                (flow_id, limit),
            )
            rows = cursor.fetchall()
        return [
            {
                "arrive_time_ms": int(row["arrive_time_ms"]),
                "direction": int(row["direction"]),
                "pkt_len": int(row["pkt_len"]),
            }
            for row in rows
        ]

    def get_latest_captured_prediction_record(self) -> dict[str, Any] | None:
        with self._lock, self._connection.cursor() as cursor:
            cursor.execute(
                """
                WITH latest_prediction AS (
                    SELECT
                        p.id AS prediction_id,
                        p.flow_id,
                        p.predicted_class,
                        p.confidence,
                        p.status,
                        p.inference_latency_ms,
                        p.device,
                        p.predicted_at,
                        p.distribution_json,
                        p.actual_label,
                        f.duration_ms,
                        f.packet_count
                    FROM predictions p
                    INNER JOIN flows f ON f.flow_id = p.flow_id
                    WHERE f.origin = 'captured'
                    ORDER BY p.predicted_at DESC
                    LIMIT 1
                )
                SELECT
                    latest_prediction.prediction_id,
                    latest_prediction.flow_id,
                    latest_prediction.predicted_class,
                    latest_prediction.confidence,
                    latest_prediction.status,
                    latest_prediction.inference_latency_ms,
                    latest_prediction.device,
                    latest_prediction.predicted_at,
                    latest_prediction.distribution_json,
                    latest_prediction.actual_label,
                    latest_prediction.duration_ms,
                    latest_prediction.packet_count,
                    last_packet.direction AS latest_direction,
                    last_packet.pkt_len AS latest_packet_size,
                    COALESCE(last_packet.arrive_time_ms - prev_packet.arrive_time_ms, last_packet.arrive_time_ms, 0) AS latest_iat_ms
                FROM latest_prediction
                LEFT JOIN LATERAL (
                    SELECT arrive_time_ms, direction, pkt_len
                    FROM flow_packets
                    WHERE flow_id = latest_prediction.flow_id
                    ORDER BY arrive_time_ms DESC
                    LIMIT 1
                ) AS last_packet ON TRUE
                LEFT JOIN LATERAL (
                    SELECT arrive_time_ms
                    FROM flow_packets
                    WHERE flow_id = latest_prediction.flow_id
                    ORDER BY arrive_time_ms DESC
                    OFFSET 1 LIMIT 1
                ) AS prev_packet ON TRUE
                """
            )
            row = cursor.fetchone()
        if row is None:
            return None
        data = _normalize_row(row)
        distribution = data.pop("distribution_json")
        data["distribution"] = distribution if isinstance(distribution, dict) else json.loads(distribution)
        class_names = data["distribution"].keys()
        predicted_class = str(data["predicted_class"])
        data["class_id"] = list(class_names).index(predicted_class) if predicted_class in class_names else -1
        return data

    def purge_runtime_data(self, *, delete_audit_logs: bool = True) -> dict[str, int]:
        deleted: dict[str, int] = {
            "flow_packets": 0,
            "predictions": 0,
            "flows": 0,
            "capture_sessions": 0,
            "audit_logs": 0,
        }
        with self._lock, self._connection.cursor() as cursor:
            cursor.execute("DELETE FROM flow_packets")
            deleted["flow_packets"] = cursor.rowcount
            cursor.execute("DELETE FROM predictions")
            deleted["predictions"] = cursor.rowcount
            cursor.execute("DELETE FROM flows")
            deleted["flows"] = cursor.rowcount
            cursor.execute("DELETE FROM capture_sessions")
            deleted["capture_sessions"] = cursor.rowcount
            if delete_audit_logs:
                cursor.execute("DELETE FROM audit_logs")
                deleted["audit_logs"] = cursor.rowcount
        return deleted

    def cleanup_expired_data(self) -> dict[str, int]:
        cutoff_dt = utc_now() - timedelta(days=self.settings.data_retention_days)
        deleted_files = 0
        deleted_capture_sessions = 0
        deleted_predictions = 0
        deleted_logs = 0

        with self._lock, self._connection.cursor() as cursor:
            cursor.execute(
                "SELECT id, packet_csv, flow_metadata_csv, summary_json FROM capture_sessions WHERE started_at < %s",
                (cutoff_dt,),
            )
            old_captures = cursor.fetchall()
            capture_ids = [int(row["id"]) for row in old_captures]
            if capture_ids:
                cursor.execute(
                    "SELECT flow_id FROM flows WHERE origin = 'captured' AND session_id = ANY(%s)",
                    (capture_ids,),
                )
                flow_ids = [int(row["flow_id"]) for row in cursor.fetchall()]
                if flow_ids:
                    cursor.execute("DELETE FROM flow_packets WHERE flow_id = ANY(%s)", (flow_ids,))
                    cursor.execute("DELETE FROM predictions WHERE flow_id = ANY(%s)", (flow_ids,))
                    deleted_predictions += cursor.rowcount
                    cursor.execute("DELETE FROM flows WHERE flow_id = ANY(%s)", (flow_ids,))
                cursor.execute("DELETE FROM capture_sessions WHERE id = ANY(%s)", (capture_ids,))
                deleted_capture_sessions += cursor.rowcount

            cursor.execute("DELETE FROM predictions WHERE predicted_at < %s", (cutoff_dt,))
            deleted_predictions += cursor.rowcount
            cursor.execute("DELETE FROM audit_logs WHERE timestamp < %s", (cutoff_dt,))
            deleted_logs = cursor.rowcount

        for capture in old_captures:
            for key in ("packet_csv", "flow_metadata_csv", "summary_json"):
                path_value = capture[key]
                if not path_value:
                    continue
                file_path = Path(str(path_value))
                try:
                    if file_path.exists():
                        file_path.unlink()
                        deleted_files += 1
                except OSError:
                    continue

        return {
            "deleted_capture_sessions": deleted_capture_sessions,
            "deleted_predictions": deleted_predictions,
            "deleted_logs": deleted_logs,
            "deleted_files": deleted_files,
        }