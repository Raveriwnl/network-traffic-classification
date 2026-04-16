from __future__ import annotations

import base64
import hashlib
import hmac
import uuid
from datetime import datetime, timedelta, timezone

import jwt

from backend.app.core.errors import AppError


ALGORITHM = "HS256"


def hash_password(password: str, salt: str) -> str:
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 120_000)
    return base64.b64encode(digest).decode("ascii")


def verify_password(candidate: str, password_hash: str, salt: str) -> bool:
    return hmac.compare_digest(hash_password(candidate, salt), password_hash)


def create_access_token(
    *,
    username: str,
    role: str,
    display_name: str,
    secret: str,
    expire_minutes: int,
) -> str:
    issued_at = datetime.now(timezone.utc)
    payload = {
        "sub": username,
        "role": role,
        "display_name": display_name,
        "jti": uuid.uuid4().hex,
        "iat": int(issued_at.timestamp()),
        "exp": int((issued_at + timedelta(minutes=expire_minutes)).timestamp()),
    }
    return jwt.encode(payload, secret, algorithm=ALGORITHM)


def decode_access_token(token: str, secret: str) -> dict[str, object]:
    try:
        payload = jwt.decode(token, secret, algorithms=[ALGORITHM])
    except jwt.PyJWTError as exc:
        raise AppError(401, "Invalid or expired token.", "AUTH_TOKEN_INVALID") from exc
    if not isinstance(payload, dict) or not payload.get("sub"):
        raise AppError(401, "Invalid or expired token.", "AUTH_TOKEN_INVALID")
    return payload