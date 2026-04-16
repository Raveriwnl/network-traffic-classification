from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
LOCAL_CONNECTION_CONFIG_PATH = PROJECT_ROOT / "backend" / "local.connection.json"


def _load_local_connection_config() -> dict[str, object]:
    if not LOCAL_CONNECTION_CONFIG_PATH.exists():
        return {}
    with LOCAL_CONNECTION_CONFIG_PATH.open("r", encoding="utf-8") as file_obj:
        data = json.load(file_obj)
    return data if isinstance(data, dict) else {}


def _coalesce(*values: object) -> object | None:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def _as_bool(value: object, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


@dataclass(slots=True)
class Settings:
    jwt_secret: str
    jwt_expire_minutes: int
    auth_salt: str
    telemetry_interval_sec: float
    postgres_dsn: str
    elasticsearch_url: str | None
    elasticsearch_index: str
    elasticsearch_username: str | None
    elasticsearch_password: str | None
    cleanup_interval_sec: float
    data_retention_days: int
    other_traffic_threshold: float
    auto_capture_enabled: bool
    auto_capture_iface: str
    auto_capture_bpf_filter: str
    auto_capture_idle_timeout: int
    auto_capture_min_packets: int
    auto_capture_label: str
    capture_flush_interval_sec: float
    capture_output_dir: Path
    capture_duration_sec: float
    capture_stop_timeout_sec: float
    model_path: Path
    huawei_features_path: Path
    huawei_metadata_path: Path
    huawei_schema_path: Path
    frontend_origins: tuple[str, ...]
    model_name: str
    model_version: str
    window_ms: int
    bins: int
    feature_count: int

    @classmethod
    def from_env(cls) -> "Settings":
        local_config = _load_local_connection_config()
        postgres_config = local_config.get("postgres") if isinstance(local_config.get("postgres"), dict) else {}
        elasticsearch_config = local_config.get("elasticsearch") if isinstance(local_config.get("elasticsearch"), dict) else {}
        capture_config = local_config.get("capture") if isinstance(local_config.get("capture"), dict) else {}
        origins_raw = os.getenv(
            "BACKEND_FRONTEND_ORIGINS",
            "http://127.0.0.1:5173,http://localhost:5173",
        )
        origins = tuple(item.strip() for item in origins_raw.split(",") if item.strip())
        model_path = Path(
            os.getenv(
                "BACKEND_MODEL_PATH",
                str(PROJECT_ROOT / "huawei_cnn_bimobilemamba_attention_best_0.9271.pt"),
            )
        )
        postgres_dsn = _coalesce(
            postgres_config.get("dsn") if isinstance(postgres_config, dict) else None,
            os.getenv("BACKEND_POSTGRES_DSN"),
        )
        if postgres_dsn is None and isinstance(postgres_config, dict):
            host = _coalesce(postgres_config.get("host"), "127.0.0.1")
            port = _coalesce(postgres_config.get("port"), 5432)
            database = _coalesce(postgres_config.get("database"), "network_traffic_classification")
            username = _coalesce(postgres_config.get("username"), "postgres")
            password = _coalesce(postgres_config.get("password"), "postgres")
            postgres_dsn = f"postgresql://{username}:{password}@{host}:{port}/{database}"
        if postgres_dsn is None:
            postgres_dsn = "postgresql://postgres:postgres@127.0.0.1:5432/network_traffic_classification"

        elasticsearch_url = _coalesce(
            elasticsearch_config.get("url") if isinstance(elasticsearch_config, dict) else None,
            os.getenv("BACKEND_ELASTICSEARCH_URL"),
            "http://127.0.0.1:9200",
        )
        elasticsearch_index = _coalesce(
            elasticsearch_config.get("index") if isinstance(elasticsearch_config, dict) else None,
            os.getenv("BACKEND_ELASTICSEARCH_INDEX"),
            "model-inference-logs",
        )
        elasticsearch_username = _coalesce(
            elasticsearch_config.get("username") if isinstance(elasticsearch_config, dict) else None,
            os.getenv("BACKEND_ELASTICSEARCH_USERNAME"),
        )
        elasticsearch_password = _coalesce(
            elasticsearch_config.get("password") if isinstance(elasticsearch_config, dict) else None,
            os.getenv("BACKEND_ELASTICSEARCH_PASSWORD"),
        )
        return cls(
            jwt_secret=os.getenv("BACKEND_JWT_SECRET", "edge-traffic-demo-secret"),
            jwt_expire_minutes=int(os.getenv("BACKEND_JWT_EXPIRE_MINUTES", "480")),
            auth_salt=os.getenv("BACKEND_AUTH_SALT", "edge-traffic-service"),
            telemetry_interval_sec=float(os.getenv("BACKEND_TELEMETRY_INTERVAL_SEC", "1.0")),
            postgres_dsn=str(postgres_dsn),
            elasticsearch_url=None if elasticsearch_url is None else str(elasticsearch_url),
            elasticsearch_index=str(elasticsearch_index),
            elasticsearch_username=None if elasticsearch_username is None else str(elasticsearch_username),
            elasticsearch_password=None if elasticsearch_password is None else str(elasticsearch_password),
            cleanup_interval_sec=float(os.getenv("BACKEND_CLEANUP_INTERVAL_SEC", "600")),
            data_retention_days=int(os.getenv("BACKEND_DATA_RETENTION_DAYS", "30")),
            other_traffic_threshold=float(os.getenv("BACKEND_OTHER_TRAFFIC_THRESHOLD", "0.45")),
            auto_capture_enabled=_as_bool(
                _coalesce(capture_config.get("enabled") if isinstance(capture_config, dict) else None, os.getenv("BACKEND_AUTO_CAPTURE_ENABLED")),
                True,
            ),
            auto_capture_iface=str(
                capture_config.get("iface")
                if isinstance(capture_config, dict) and "iface" in capture_config
                else os.getenv("BACKEND_AUTO_CAPTURE_IFACE", "")
            ),
            auto_capture_bpf_filter=str(
                _coalesce(
                    capture_config.get("bpf_filter") if isinstance(capture_config, dict) else None,
                    os.getenv("BACKEND_AUTO_CAPTURE_BPF_FILTER"),
                    "tcp or udp",
                )
            ),
            auto_capture_idle_timeout=int(
                _coalesce(
                    capture_config.get("idle_timeout") if isinstance(capture_config, dict) else None,
                    os.getenv("BACKEND_AUTO_CAPTURE_IDLE_TIMEOUT"),
                    5,
                )
            ),
            auto_capture_min_packets=int(
                _coalesce(
                    capture_config.get("min_packets") if isinstance(capture_config, dict) else None,
                    os.getenv("BACKEND_AUTO_CAPTURE_MIN_PACKETS"),
                    1,
                )
            ),
            auto_capture_label=str(
                _coalesce(
                    capture_config.get("label") if isinstance(capture_config, dict) else None,
                    os.getenv("BACKEND_AUTO_CAPTURE_LABEL"),
                    "auto_capture",
                )
            ),
            capture_flush_interval_sec=float(
                _coalesce(
                    capture_config.get("flush_interval_sec") if isinstance(capture_config, dict) else None,
                    os.getenv("BACKEND_CAPTURE_FLUSH_INTERVAL_SEC"),
                    1.0,
                )
            ),
            capture_output_dir=Path(
                os.getenv(
                    "BACKEND_CAPTURE_OUTPUT_DIR",
                    str(PROJECT_ROOT / "datasets" / "raw" / "mydata" / "captures"),
                )
            ),
            capture_duration_sec=float(os.getenv("BACKEND_CAPTURE_DURATION_SEC", "86400")),
            capture_stop_timeout_sec=float(os.getenv("BACKEND_CAPTURE_STOP_TIMEOUT_SEC", "10.0")),
            model_path=model_path,
            huawei_features_path=PROJECT_ROOT / "datasets" / "processed" / "huawei" / "huawei_5s_1000bins_features.npz",
            huawei_metadata_path=PROJECT_ROOT / "datasets" / "processed" / "huawei" / "huawei_5s_1000bins_metadata.csv",
            huawei_schema_path=PROJECT_ROOT / "datasets" / "processed" / "huawei" / "huawei_5s_1000bins_schema.json",
            frontend_origins=origins,
            model_name="cnn_bimobilemamba_attention",
            model_version=datetime.now(timezone.utc).strftime("%Y.%m.%d"),
            window_ms=5000,
            bins=1000,
            feature_count=10,
        )