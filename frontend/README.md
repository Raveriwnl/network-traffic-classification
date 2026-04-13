# Vue3 Frontend for Edge Traffic Classification

## Features

- Vue 3 + Vite architecture
- JWT 登录页与本地令牌持久化
- 前端路由守卫与管理员权限控制
- Realtime telemetry via authenticated WebSocket
- 管理员运行日志查看界面
- Line chart for `packet_size` and `iat`
- Pie chart for 8-class traffic distribution
- System panel for accuracy, recall, latency, power, and throughput
- Recent packet metadata table (`packet_size`, `iat`, `direction`)

## 1) Start FastAPI simulator

```bash
cd frontend/backend_sim
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Runs on `http://127.0.0.1:8000` and exposes `ws://127.0.0.1:8000/ws/telemetry`.

Built-in demo accounts:

- `admin / admin123`
- `analyst / traffic123`

## 2) Start frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5173`.

## Optional env

Create `.env` in `frontend/` to override API or websocket url:

```bash
VITE_API_BASE_URL=http://127.0.0.1:8000
VITE_WS_URL=ws://127.0.0.1:8000/ws/telemetry
```

如果设置了 `VITE_WS_URL`，前端仍会自动在连接时追加 JWT `token` 参数。
