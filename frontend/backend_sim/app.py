import asyncio
import hashlib
import os
import random
import uuid
from collections import deque
from datetime import datetime, timedelta, timezone

import jwt
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title='Edge Traffic Auth Simulator', version='0.2.0')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://127.0.0.1:5173', 'http://localhost:5173'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

CLASS_NAMES = [
    'openlive',
    'live',
    'message',
    'short_video',
    'video',
    'meeting',
    'phone_game',
    'cloud_game',
]

SECRET_KEY = os.getenv('JWT_SECRET', 'edge-traffic-demo-secret')
ALGORITHM = 'HS256'
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv('JWT_EXPIRE_MINUTES', '480'))
PASSWORD_SALT = os.getenv('AUTH_SALT', 'edge-traffic-simulator')
LOG_BUFFER_SIZE = int(os.getenv('LOG_BUFFER_SIZE', '500'))

LOGS = deque(maxlen=LOG_BUFFER_SIZE)


def hash_password(password: str) -> str:
    return hashlib.sha256(f'{PASSWORD_SALT}:{password}'.encode('utf-8')).hexdigest()


USERS = {
    'admin': {
        'username': 'admin',
        'display_name': 'System Administrator',
        'role': 'admin',
        'password_hash': hash_password('admin123'),
    },
    'analyst': {
        'username': 'analyst',
        'display_name': 'Traffic Analyst',
        'role': 'analyst',
        'password_hash': hash_password('traffic123'),
    },
}


class LoginRequest(BaseModel):
    username: str
    password: str


def build_public_user(user: dict) -> dict:
    return {
        'username': user['username'],
        'display_name': user['display_name'],
        'role': user['role'],
    }


def get_client_ip(request: Request | None = None, websocket: WebSocket | None = None) -> str:
    client = None
    if request is not None:
        client = request.client
    elif websocket is not None:
        client = websocket.client
    return client.host if client else 'unknown'


def append_log(level: str, action: str, message: str, actor: str | None = None, role: str | None = None, ip: str | None = None) -> None:
    LOGS.appendleft(
        {
            'id': str(uuid.uuid4()),
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'level': level,
            'action': action,
            'message': message,
            'actor': actor,
            'role': role,
            'ip': ip,
        }
    )


def maybe_emit_system_log() -> None:
    if random.random() >= 0.18:
        return

    message = random.choice(
        [
            'Telemetry stream stable.',
            'Model inference queue drained successfully.',
            'Frontend heartbeat acknowledged.',
            'Classifier service warming cache window.',
        ]
    )
    append_log('info', 'system_heartbeat', message, actor='system', role='service', ip='127.0.0.1')


def create_access_token(user: dict) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        'sub': user['username'],
        'role': user['role'],
        'display_name': user['display_name'],
        'iat': now,
        'exp': now + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.PyJWTError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Invalid or expired token.') from exc


def verify_password(candidate: str, user: dict) -> bool:
    return hash_password(candidate) == user['password_hash']


def make_distribution() -> dict:
    weights = [random.random() for _ in CLASS_NAMES]
    total = sum(weights)
    return {name: round(weight / total, 4) for name, weight in zip(CLASS_NAMES, weights)}


def make_payload() -> dict:
    maybe_emit_system_log()
    distribution = make_distribution()
    predicted = max(distribution, key=distribution.get)

    return {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'stream': {
            'packet_size': random.randint(60, 1500),
            'iat': round(random.uniform(0.2, 180.0), 3),
            'direction': random.choice(['uplink', 'downlink']),
        },
        'metrics': {
            'accuracy': round(random.uniform(0.9, 0.98), 4),
            'recall': round(random.uniform(0.89, 0.97), 4),
            'inference_latency_ms': round(random.uniform(2.5, 18.0), 3),
            'power_w': round(random.uniform(2.2, 9.8), 3),
            'flows_per_sec': round(random.uniform(100, 280), 2),
        },
        'prediction': {
            'class_id': CLASS_NAMES.index(predicted),
            'class_name': predicted,
            'confidence': round(distribution[predicted], 4),
        },
        'distribution': distribution,
    }


def get_current_user(request: Request) -> dict:
    authorization = request.headers.get('Authorization', '')
    scheme, _, token = authorization.partition(' ')

    if scheme.lower() != 'bearer' or not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Missing bearer token.')

    payload = decode_token(token)
    username = payload.get('sub')
    user = USERS.get(username)

    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Unknown user.')

    return build_public_user(user)


def require_admin(user: dict = Depends(get_current_user)) -> dict:
    if user['role'] != 'admin':
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail='Administrator role required.')
    return user


append_log('info', 'service_boot', 'FastAPI simulator initialized.', actor='system', role='service', ip='127.0.0.1')


@app.get('/api/health')
def health() -> dict:
    return {
        'status': 'ok',
        'service': 'fastapi-backend-sim',
        'auth': 'jwt',
        'users': [build_public_user(user) for user in USERS.values()],
    }


@app.post('/api/auth/login')
def login(payload: LoginRequest, request: Request) -> dict:
    ip = get_client_ip(request=request)
    user = USERS.get(payload.username)

    if not user or not verify_password(payload.password, user):
        append_log('warn', 'login_failed', 'Login failed due to invalid credentials.', actor=payload.username, role='unknown', ip=ip)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Invalid username or password.')

    token = create_access_token(user)
    public_user = build_public_user(user)
    append_log('info', 'login_success', 'User authenticated successfully.', actor=user['username'], role=user['role'], ip=ip)

    return {
        'access_token': token,
        'token_type': 'bearer',
        'user': public_user,
    }


@app.get('/api/auth/me')
def me(request: Request, user: dict = Depends(get_current_user)) -> dict:
    append_log('info', 'profile_check', 'Frontend refreshed active profile.', actor=user['username'], role=user['role'], ip=get_client_ip(request=request))
    return user


@app.get('/api/telemetry/latest')
def telemetry_latest(request: Request, user: dict = Depends(get_current_user)) -> dict:
    append_log('info', 'telemetry_snapshot', 'Frontend requested latest telemetry snapshot.', actor=user['username'], role=user['role'], ip=get_client_ip(request=request))
    return make_payload()


@app.get('/api/admin/logs')
def admin_logs(
    request: Request,
    limit: int = Query(default=120, ge=1, le=200),
    admin: dict = Depends(require_admin),
) -> dict:
    append_log('info', 'admin_logs_view', 'Administrator opened the runtime log viewer.', actor=admin['username'], role=admin['role'], ip=get_client_ip(request=request))
    return {'logs': list(LOGS)[:limit]}


@app.websocket('/ws/telemetry')
async def telemetry_stream(websocket: WebSocket) -> None:
    token = websocket.query_params.get('token')

    if not token:
      await websocket.accept()
      await websocket.close(code=4401, reason='Missing token.')
      return

    try:
        payload = decode_token(token)
    except HTTPException:
        await websocket.accept()
        await websocket.close(code=4401, reason='Invalid token.')
        return

    username = payload.get('sub')
    user = USERS.get(username)
    if not user:
        await websocket.accept()
        await websocket.close(code=4401, reason='Unknown user.')
        return

    await websocket.accept()
    ip = get_client_ip(websocket=websocket)
    append_log('info', 'ws_connected', 'Authenticated telemetry websocket connected.', actor=user['username'], role=user['role'], ip=ip)

    try:
        while True:
            await websocket.send_json(make_payload())
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        append_log('warn', 'ws_disconnected', 'Telemetry websocket disconnected.', actor=user['username'], role=user['role'], ip=ip)


if __name__ == '__main__':
    uvicorn.run('app:app', host='0.0.0.0', port=8000, reload=True)
