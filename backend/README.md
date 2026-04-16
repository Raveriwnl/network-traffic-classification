# Backend

This backend implements the documented FastAPI service for the frontend dashboard and admin pages.

## Features

- JWT login with `admin` and `analyst` roles
- `/api/auth/login`, `/api/auth/me`, `/api/telemetry/latest`, `/api/admin/logs`
- `/ws/telemetry?token=<jwt>` realtime telemetry stream
- PostgreSQL-backed user, flow, prediction, log, and capture storage
- Elasticsearch-backed model inference logs
- PyTorch inference using `huawei_cnn_bimobilemamba_attention_best_0.9271.pt`
- Capture start/stop endpoints backed by `data_collection/traffic_collector.py`
- Low-confidence predictions are labeled as `other`

## Run

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```

Before startup, ensure PostgreSQL and Elasticsearch are reachable. The default settings assume PostgreSQL on `127.0.0.1:5432` and Elasticsearch on `127.0.0.1:9200`.

The backend also checks `backend/local.connection.json` first for PostgreSQL and Elasticsearch connection settings. That file is ignored by git and is intended for local secrets.

The same local file can also define startup capture behavior. By default, the backend now auto-starts a capture session on boot unless you disable it with `capture.enabled = false` or `BACKEND_AUTO_CAPTURE_ENABLED=false`.

Live capture is flushed to PostgreSQL incrementally. The default behavior is to refresh the current captured-flow snapshot every 1 second instead of waiting until the capture session stops.

On Windows, live capture requires Npcap and a valid interface name such as `WLAN` or `以太网`. Using `any` is supported by the collector, but an explicit interface is more stable on Windows.

## Environment variables

- `BACKEND_JWT_SECRET`
- `BACKEND_JWT_EXPIRE_MINUTES`
- `BACKEND_AUTH_SALT`
- `BACKEND_TELEMETRY_INTERVAL_SEC`
- `BACKEND_MODEL_PATH`
- `BACKEND_FRONTEND_ORIGINS`
- `BACKEND_POSTGRES_DSN`
- `BACKEND_ELASTICSEARCH_URL`
- `BACKEND_ELASTICSEARCH_INDEX`
- `BACKEND_ELASTICSEARCH_USERNAME`
- `BACKEND_ELASTICSEARCH_PASSWORD`
- `BACKEND_CLEANUP_INTERVAL_SEC`
- `BACKEND_DATA_RETENTION_DAYS`
- `BACKEND_OTHER_TRAFFIC_THRESHOLD`
- `BACKEND_AUTO_CAPTURE_ENABLED`
- `BACKEND_AUTO_CAPTURE_IFACE`
- `BACKEND_AUTO_CAPTURE_BPF_FILTER`
- `BACKEND_AUTO_CAPTURE_IDLE_TIMEOUT`
- `BACKEND_AUTO_CAPTURE_MIN_PACKETS`
- `BACKEND_AUTO_CAPTURE_LABEL`
- `BACKEND_CAPTURE_OUTPUT_DIR`
- `BACKEND_CAPTURE_FLUSH_INTERVAL_SEC`
- `BACKEND_CAPTURE_DURATION_SEC`
- `BACKEND_CAPTURE_STOP_TIMEOUT_SEC`