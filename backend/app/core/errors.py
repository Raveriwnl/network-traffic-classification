from __future__ import annotations


class AppError(Exception):
    def __init__(self, status_code: int, detail: str, code: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail
        self.code = code


def build_error_payload(detail: str, code: str, request_id: str) -> dict[str, str]:
    return {
        "detail": detail,
        "code": code,
        "request_id": request_id,
    }