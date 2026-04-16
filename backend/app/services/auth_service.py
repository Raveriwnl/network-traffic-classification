from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from backend.app.core.config import Settings
from backend.app.core.errors import AppError
from backend.app.core.security import create_access_token, decode_access_token, hash_password, verify_password
from backend.app.services.audit_service import AuditService
from backend.app.services.database_service import DatabaseService


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(slots=True)
class UserRecord:
    id: int
    username: str
    display_name: str
    role: str
    status: str
    password_hash: str
    created_at: str
    updated_at: str
    last_login_at: str | None = None

    def public_dict(self) -> dict[str, str]:
        return {
            "username": self.username,
            "display_name": self.display_name,
            "role": self.role,
        }

    def admin_dict(self) -> dict[str, str | int | None]:
        data: dict[str, str | int | None] = {
            "id": self.id,
            "username": self.username,
            "display_name": self.display_name,
            "role": self.role,
            "status": self.status,
            "last_login_at": self.last_login_at,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        return data


class AuthService:
    def __init__(self, settings: Settings, audit_service: AuditService, database_service: DatabaseService) -> None:
        self.settings = settings
        self.audit_service = audit_service
        self.database_service = database_service
        created_at = iso_now()
        self.database_service.ensure_default_users(
            [
                {
                    "username": "admin",
                    "display_name": "System Administrator",
                    "role": "admin",
                    "status": "active",
                    "password_hash": hash_password("admin123", settings.auth_salt),
                    "created_at": created_at,
                    "updated_at": created_at,
                },
                {
                    "username": "analyst",
                    "display_name": "Traffic Analyst",
                    "role": "analyst",
                    "status": "active",
                    "password_hash": hash_password("traffic123", settings.auth_salt),
                    "created_at": created_at,
                    "updated_at": created_at,
                },
            ]
        )

    @staticmethod
    def _to_user(row: dict[str, object]) -> UserRecord:
        return UserRecord(
            id=int(row["id"]),
            username=str(row["username"]),
            display_name=str(row["display_name"]),
            role=str(row["role"]),
            status=str(row["status"]),
            password_hash=str(row["password_hash"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            last_login_at=None if row.get("last_login_at") is None else str(row["last_login_at"]),
        )

    def login(self, username: str, password: str, ip: str, user_agent: str | None = None) -> dict[str, object]:
        row = self.database_service.get_user_by_username(username)
        user = None if row is None else self._to_user(row)
        if not user or not verify_password(password, user.password_hash, self.settings.auth_salt):
            self.audit_service.append(
                level="warn",
                action="login_failed",
                message="Login failed due to invalid credentials.",
                actor=username,
                role="unknown",
                ip=ip,
            )
            raise AppError(401, "Invalid username or password.", "AUTH_LOGIN_FAILED")
        if user.status != "active":
            self.audit_service.append(
                level="warn",
                action="login_blocked",
                message="Disabled account attempted to log in.",
                actor=user.username,
                role=user.role,
                ip=ip,
            )
            raise AppError(423, "Account is disabled.", "AUTH_ACCOUNT_DISABLED")
        user.last_login_at = iso_now()
        user.updated_at = iso_now()
        self.database_service.update_user_login(username, last_login_at=user.last_login_at, updated_at=user.updated_at)
        token = create_access_token(
            username=user.username,
            role=user.role,
            display_name=user.display_name,
            secret=self.settings.jwt_secret,
            expire_minutes=self.settings.jwt_expire_minutes,
        )
        self.audit_service.append(
            level="info",
            action="login_success",
            message="User authenticated successfully.",
            actor=user.username,
            role=user.role,
            ip=ip,
        )
        return {
            "access_token": token,
            "token_type": "bearer",
            "user": user.public_dict(),
        }

    def authenticate_token(self, token: str) -> UserRecord:
        payload = decode_access_token(token, self.settings.jwt_secret)
        username = str(payload["sub"])
        row = self.database_service.get_user_by_username(username)
        user = None if row is None else self._to_user(row)
        if not user:
            raise AppError(401, "Invalid or expired token.", "AUTH_TOKEN_INVALID")
        if user.status != "active":
            raise AppError(423, "Account is disabled.", "AUTH_ACCOUNT_DISABLED")
        return user

    def get_user_by_id(self, user_id: int) -> UserRecord:
        row = self.database_service.get_user_by_id(user_id)
        if row is None:
            raise AppError(404, "User not found.", "USER_NOT_FOUND")
        return self._to_user(row)

    def list_users(self) -> list[dict[str, str | int | None]]:
        return [self._to_user(row).admin_dict() for row in self.database_service.list_users()]

    def create_user(self, username: str, password: str, display_name: str, role: str) -> dict[str, str | int | None]:
        if self.database_service.get_user_by_username(username) is not None:
            raise AppError(409, "Username already exists.", "USER_ALREADY_EXISTS")
        created_at = iso_now()
        row = self.database_service.create_user(
            {
                "username": username,
                "display_name": display_name,
                "role": role,
                "status": "active",
                "password_hash": hash_password(password, self.settings.auth_salt),
                "created_at": created_at,
                "updated_at": created_at,
            }
        )
        return self._to_user(row).admin_dict()

    def update_user(
        self,
        user_id: int,
        *,
        display_name: str | None,
        role: str | None,
        status: str | None,
    ) -> dict[str, str | int | None]:
        row = self.database_service.update_user(
            user_id,
            display_name=display_name,
            role=role,
            status=status,
            updated_at=iso_now(),
        )
        if row is None:
            raise AppError(404, "User not found.", "USER_NOT_FOUND")
        return self._to_user(row).admin_dict()