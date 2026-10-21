"""Public demo controls. Secrets and client IPs are never returned or persisted."""
from __future__ import annotations
import hashlib
import hmac
import os
import secrets
import sqlite3
import time
from datetime import datetime, timezone
from urllib.parse import urlsplit

from .models import UserFacingError


def public_mode():
    return os.getenv("SCENE_PUBLIC", "0") == "1"


def public_url():
    return os.getenv("APP_PUBLIC_URL", os.getenv("RENDER_EXTERNAL_URL", "")).rstrip("/")


def allowed_hosts():
    hosts = {"127.0.0.1", "localhost", "::1", "testserver"}
    if public_mode():
        host = urlsplit(public_url()).hostname
        if host:
            hosts.add(host)
        hosts.update(h.strip() for h in os.getenv("APP_ALLOWED_HOSTS", "").split(",") if h.strip() and "*" not in h)
    return hosts


class VisitorSessions:
    def __init__(self):
        self.secret = os.getenv("SESSION_SECRET", "").encode() or secrets.token_bytes(32)

    def issue(self, cookie):
        if cookie and "." in cookie:
            value, signature = cookie.rsplit(".", 1)
            expected = hmac.new(self.secret, value.encode(), hashlib.sha256).hexdigest()
            if len(value) == 32 and hmac.compare_digest(signature, expected):
                return value, cookie, False
        value = secrets.token_hex(16)
        signature = hmac.new(self.secret, value.encode(), hashlib.sha256).hexdigest()
        return value, value + "." + signature, True


class PublicQuota:
    def __init__(self, root):
        self.path = root / "data/public_usage.sqlite3"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS counters(k TEXT PRIMARY KEY, n INTEGER NOT NULL)")

    def connect(self):
        return sqlite3.connect(self.path, timeout=30)

    def take(self, key, maximum, message):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT n FROM counters WHERE k=?", (key,)).fetchone()
            count = row[0] if row else 0
            if count >= maximum:
                raise UserFacingError(message)
            db.execute("INSERT OR REPLACE INTO counters VALUES(?,?)", (key, count + 1))

    def before_model_call(self):
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        limit = max(1, int(os.getenv("GMS_DAILY_CALL_LIMIT", "100")))
        self.take("gms:" + day, limit, "오늘의 공개 검색·답변 사용량을 모두 사용했습니다. 영화 목록·비교 기록·PDF 자료를 열람하거나 내일 다시 방문해 주세요.")

    def request(self, client_ip, use_gms):
        # Hash IP addresses before persistence. A new cookie does not reset the IP quota.
        identity = hashlib.sha256(client_ip.encode()).hexdigest()
        bucket = int(time.time()) // 600
        self.take(f"request:{identity}:{bucket}", 5, "짧은 시간에 요청이 많습니다. 잠시 후 다시 이용해 주세요.")
        if use_gms:
            hour = int(time.time()) // 3600
            self.take(f"visitor-gms:{identity}:{hour}", int(os.getenv("GMS_HOURLY_REQUEST_LIMIT", "3")),
                      "공개 검색·답변은 접속 환경별 시간당 3회까지 가능합니다. 영화 목록·비교 기록·PDF 자료는 계속 열람할 수 있습니다.")
