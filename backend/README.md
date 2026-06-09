# Backend

当前后端是一个基于 FastAPI 的实时加密流量分类服务，负责鉴权、遥测推送、管理员审计、用户管理、抓包会话管理，以及基于 PyTorch 模型的在线推理。

## 当前能力概览

- JWT 鉴权，内置 `admin` 和 `analyst` 两种角色。
- 基于 PostgreSQL 持久化用户、审计日志、抓包会话、流记录、包序列和预测结果。
- 启动时自动初始化数据库表结构，并在库中不存在时写入默认账号。
- 加载根目录模型权重 `huawei_cnn_bimobilemamba_attention_best_0.9271.pt` 进行推理。
- 从 `datasets/processed/huawei/huawei_5s_1000bins_schema.json` 读取类别定义。
- 通过 `/api/telemetry/latest` 和 `/ws/telemetry?token=<jwt>` 向前端提供最新遥测快照。
- 提供管理员日志查询、用户管理、抓包启动/停止、流量明细和最近预测记录接口。
- 当模型最高置信度低于 `BACKEND_OTHER_TRAFFIC_THRESHOLD` 时，将预测类别降级为 `other`。

## 主要接口

### 公共接口

- `GET /api/health`: 健康检查，返回模型名、模型版本和类别列表。

### 鉴权接口

- `POST /api/auth/login`: 用户登录，返回 JWT 和当前用户信息。
- `GET /api/auth/me`: 校验并刷新当前登录用户信息。

### 遥测接口

- `GET /api/telemetry/latest`: 返回最新一帧遥测数据。
- `GET /ws/telemetry?token=<jwt>`: WebSocket 实时推送遥测数据。

遥测数据包含以下几部分：

- `stream`: 当前流的 `flow_id`、`packet_size`、`iat`、`direction`、`duration_ms`、`packet_count`
- `metrics`: `accuracy`、`recall`、`inference_latency_ms`、`power_w`、`flows_per_sec`
- `prediction`: `class_id`、`class_name`、`confidence`、`model_name`、`model_version`
- `distribution`: 8 类业务分布

### 管理员接口

- `GET /api/admin/logs`: 查询审计日志，支持 `limit`、`level`、`actor`、`start_time`、`end_time`。
- `GET /api/admin/users`: 获取用户列表。
- `POST /api/admin/users`: 创建用户。
- `PATCH /api/admin/users/{user_id}`: 更新显示名、角色和状态。
- `POST /api/admin/capture/start`: 启动抓包会话。
- `POST /api/admin/capture/stop`: 停止当前抓包会话。
- `GET /api/flows`: 分页查询已入库流量，支持按 IP、协议、预测类别、时间范围过滤。
- `GET /api/flows/{flow_id}`: 获取单条流量的元数据、特征摘要、包序列摘要和预测历史。
- `GET /api/predictions/latest`: 查询最近预测记录。

说明：当前前端已经接入管理员日志和用户管理页面；抓包控制、流量列表和预测历史接口目前仍主要面向后端管理或后续扩展使用。

## 运行前准备

后端启动前需要满足以下条件：

- Python 环境已安装并可用。
- PostgreSQL 可连接。
- 根目录模型权重文件存在。
- `datasets/processed/huawei/` 下的 schema 文件存在。

默认 PostgreSQL DSN 为：

```text
postgresql://postgres:postgres@127.0.0.1:5432/network_traffic_classification
```

## 启动方式

在仓库根目录执行：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```

启动后默认服务地址为 `http://127.0.0.1:8000`。

## 默认账号

首次连接数据库且 `users` 表为空时，会自动写入以下账号：

- `admin / admin123`
- `analyst / traffic123`

## 配置方式

后端同时支持环境变量和本地配置文件 `backend/local.connection.json`。该文件适合保存本机数据库连接和抓包参数，并且应保持在 git 之外。

示例：

```json
{
	"postgres": {
		"host": "127.0.0.1",
		"port": 5432,
		"database": "network_traffic_classification",
		"username": "postgres",
		"password": "postgres"
	},
	"capture": {
		"enabled": true,
		"iface": "eth0",
		"bpf_filter": "tcp or udp",
		"idle_timeout": 5,
		"min_packets": 1,
		"label": "auto_capture",
		"flush_interval_sec": 1.0
	}
}
```

## 关键环境变量

- `BACKEND_JWT_SECRET`: JWT 签名密钥。
- `BACKEND_JWT_EXPIRE_MINUTES`: Token 过期时间，默认 480 分钟。
- `BACKEND_AUTH_SALT`: 密码散列盐值。
- `BACKEND_FRONTEND_ORIGINS`: 允许的前端来源，默认包含 `http://127.0.0.1:5173` 和 `http://localhost:5173`。
- `BACKEND_POSTGRES_DSN`: PostgreSQL 连接串。
- `BACKEND_MODEL_PATH`: 模型权重路径。
- `BACKEND_OTHER_TRAFFIC_THRESHOLD`: 低置信度降级为 `other` 的阈值，默认 `0.45`。
- `BACKEND_TELEMETRY_INTERVAL_SEC`: 遥测刷新周期，默认 `1.0` 秒。
- `BACKEND_CLEANUP_INTERVAL_SEC`: 数据清理轮询周期。
- `BACKEND_DATA_RETENTION_DAYS`: 历史数据保留天数。
- `BACKEND_AUTO_CAPTURE_ENABLED`: 启动时是否自动开启抓包，默认开启。
- `BACKEND_AUTO_CAPTURE_IFACE`: 自动抓包接口名。
- `BACKEND_AUTO_CAPTURE_BPF_FILTER`: 自动抓包 BPF 过滤条件。
- `BACKEND_AUTO_CAPTURE_IDLE_TIMEOUT`: 流空闲超时秒数。
- `BACKEND_AUTO_CAPTURE_MIN_PACKETS`: 最小包数阈值。
- `BACKEND_AUTO_CAPTURE_LABEL`: 自动抓包任务标签。
- `BACKEND_CAPTURE_OUTPUT_DIR`: 抓包输出目录。
- `BACKEND_CAPTURE_FLUSH_INTERVAL_SEC`: 抓包写库刷新周期。
- `BACKEND_CAPTURE_DURATION_SEC`: 单次抓包最大持续时间。
- `BACKEND_CAPTURE_STOP_TIMEOUT_SEC`: 停止抓包时的等待超时。

## 依赖说明

`backend/requirements.txt` 当前核心依赖包括：

- `fastapi` 和 `uvicorn` 用于服务与接口暴露。
- `PyJWT` 用于 JWT 鉴权。
- `psycopg[binary]` 用于 PostgreSQL 访问。
- `torch` 用于加载和执行分类模型。
- `numpy`、`pandas` 用于数据处理。
- `scapy` 用于实时抓包。

如果在 Linux 上进行在线抓包，通常需要为抓包相关权限和网卡访问单独配置运行环境。