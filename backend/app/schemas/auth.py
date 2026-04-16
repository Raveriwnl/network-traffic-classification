from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=128)


class UserResponse(BaseModel):
    username: str
    display_name: str
    role: Literal["admin", "analyst"]


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


class UserItemResponse(UserResponse):
    id: int
    status: Literal["active", "disabled"]
    last_login_at: str | None = None
    created_at: str
    updated_at: str


class UsersResponse(BaseModel):
    users: list[UserItemResponse]


class UserCreateRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=6, max_length=128)
    display_name: str = Field(min_length=1, max_length=128)
    role: Literal["admin", "analyst"]


class UserUpdateRequest(BaseModel):
    display_name: str | None = Field(default=None, min_length=1, max_length=128)
    role: Literal["admin", "analyst"] | None = None
    status: Literal["active", "disabled"] | None = None