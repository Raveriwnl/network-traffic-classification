# Frontend

当前前端是一个基于 Vue 3 + Vite 的管理控制台，面向后端实时流量分类服务，提供登录、实时遥测看板、管理员日志和用户管理能力。

## 当前页面与功能

### 登录页

- 通过 `POST /api/auth/login` 完成登录。
- 登录成功后将 JWT 和用户信息持久化到浏览器 `localStorage`。
- 路由会根据 `redirect` 参数返回原目标页面。

### 实时监控看板

- 首次进入时调用 `GET /api/telemetry/latest` 加载最近一帧数据。
- 之后通过 `ws://.../ws/telemetry?token=<jwt>` 建立 WebSocket 实时刷新。
- 展示推理延迟、功耗、吞吐、当前主导业务等 KPI。
- 以折线图展示最近 60 秒 `packet_size` 和 `iat`。
- 以饼图展示 8 类业务实时分布。
- 展示最新分类结果、置信度、精度、召回率。
- 展示最近 18 条输入元数据摘要，包括 `packet_size`、`iat`、`direction`。
- 支持在页面中开启或关闭实时监控，并在本地记住开关状态。

### 管理员日志页

- 对接 `GET /api/admin/logs`。
- 支持按日志级别、操作者、时间范围过滤。
- 支持前端关键字检索、分页和每页条数切换。
- 默认每 5 秒自动刷新一次。

### 用户管理页

- 对接 `GET /api/admin/users`、`POST /api/admin/users`、`PATCH /api/admin/users/{user_id}`。
- 支持创建管理员和分析员账号。
- 支持编辑显示名、角色和启用状态。
- 展示最近登录时间、创建时间和账号状态统计。

## 当前权限规则

- 未登录用户只能访问登录页。
- `admin` 和 `analyst` 都可以访问实时监控看板。
- 只有 `admin` 可以访问日志页和用户管理页。
- 当 API 返回 401 或 WebSocket 以未授权状态关闭时，前端会清理本地登录态并跳转回登录页。

## 当前接入的后端接口

- `POST /api/auth/login`
- `GET /api/auth/me`
- `GET /api/telemetry/latest`
- `GET /ws/telemetry?token=<jwt>`
- `GET /api/admin/logs`
- `GET /api/admin/users`
- `POST /api/admin/users`
- `PATCH /api/admin/users/{user_id}`

说明：后端还提供抓包控制、流量列表和预测历史接口，但当前前端尚未接入对应页面。

## 启动方式

建议先启动仓库中的真实后端，再启动前端。

```bash
cd frontend
npm install
npm run dev
```

默认开发地址：`http://127.0.0.1:5173`。

Vite 当前配置为：

- Host: `0.0.0.0`
- Port: `5173`

## 环境变量

可以在 `frontend/.env` 中覆盖后端地址：

```bash
VITE_API_BASE_URL=http://127.0.0.1:8000
VITE_WS_URL=ws://127.0.0.1:8000/ws/telemetry
```

说明：

- `VITE_API_BASE_URL` 默认为 `http://127.0.0.1:8000`。
- 如果未设置 `VITE_WS_URL`，前端会根据 `VITE_API_BASE_URL` 自动推导 WebSocket 地址，并默认连接 `/ws/telemetry`。
- 无论是否显式设置 `VITE_WS_URL`，前端都会在连接时自动追加 JWT `token` 查询参数。

## 联调建议

当前前端默认面向仓库中的真实 FastAPI 后端，而不是模拟器。完整联调建议如下：

1. 在仓库根目录启动 `backend.app.main:app`。
2. 确认后端数据库、模型权重和处理后数据可正常加载。
3. 启动前端开发服务器。
4. 使用默认账号登录：`admin / admin123` 或 `analyst / traffic123`。

## 关于 `backend_sim`

`frontend/backend_sim/` 仍可用于单独演示界面或脱离真实后端做 UI 调试，但它不再代表当前系统的主接入方式。更新文档和联调说明时，应以前后端真实实现为准。
