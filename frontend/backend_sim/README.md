# Flask Backend Simulator

A lightweight mock backend for the Vue dashboard.

## Start

```bash
cd frontend/backend_sim
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

## Endpoints

- `GET /api/health`: liveness check
- `GET /api/once`: one random telemetry payload
- `WS /ws/telemetry`: push one payload per second
