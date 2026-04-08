# Vue3 Frontend for Edge Traffic Classification

## Features

- Vue 3 + Vite architecture
- Realtime telemetry via WebSocket
- Line chart for `packet_size` and `iat`
- Pie chart for 8-class traffic distribution
- System panel for accuracy, recall, latency, power, and throughput
- Recent packet metadata table (`packet_size`, `iat`, `direction`)

## 1) Start Flask simulator

```bash
cd frontend/backend_sim
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Runs on `ws://127.0.0.1:5000/ws/telemetry`.

## 2) Start frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5173`.

## Optional env

Create `.env` in `frontend/` to override websocket url:

```bash
VITE_WS_URL=ws://127.0.0.1:5000/ws/telemetry
```
