"""
Auth — shared team password + signed cookie session.
ทุกคนในทีมใช้รหัสเดียวกัน (APP_PASSWORD). เก็บชื่อผู้ใช้ที่พิมพ์ตอน login
ไว้ใน session เพื่อ attribute การ log/trade (ไม่ใช่ security boundary).
"""
import os
import hmac

from fastapi import Request
from fastapi.responses import RedirectResponse

APP_PASSWORD = os.environ.get("APP_PASSWORD", "sentinel")


def check_password(pw: str) -> bool:
    return hmac.compare_digest(pw or "", APP_PASSWORD)


def current_user(request: Request) -> str | None:
    return request.session.get("user")


def require_login(request: Request):
    """คืน RedirectResponse ถ้ายังไม่ login, ไม่งั้นคืน None"""
    if not current_user(request):
        return RedirectResponse("/login", status_code=302)
    return None
