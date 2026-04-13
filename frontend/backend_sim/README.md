# FastAPI Backend Simulator

一个用于前端联调的 FastAPI 模拟后端，支持 JWT 登录、管理员日志查询，以及带令牌认证的实时遥测 WebSocket。

## Start

```bash
cd frontend/backend_sim
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

默认启动在 `http://127.0.0.1:8000`。

## Built-in Accounts

- `admin / admin123`: 管理员，可访问日志页
- `analyst / traffic123`: 普通分析员，仅可访问实时看板

## Endpoints

- `GET /api/health`: 健康检查与账号元数据
- `POST /api/auth/login`: 登录并签发 JWT
- `GET /api/auth/me`: 刷新当前用户信息，需要 Bearer Token
- `GET /api/telemetry/latest`: 获取一次实时遥测快照，需要 Bearer Token
- `GET /api/admin/logs?limit=120`: 获取运行日志，仅管理员可访问
- `WS /ws/telemetry?token=<jwt>`: 已认证用户的实时流式遥测

## Notes

- JWT 默认有效期 8 小时，可通过 `JWT_EXPIRE_MINUTES` 覆盖。
- 可通过 `JWT_SECRET` 和 `AUTH_SALT` 覆盖默认密钥。
