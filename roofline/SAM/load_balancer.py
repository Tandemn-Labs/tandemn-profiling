#!/usr/bin/env python3
"""
Simple router:
- If SkyServe is READY -> forward request to SkyServe
- Else -> forward request to Cloud Run
- While NOT READY, every request forwarded to Cloud Run also "pokes" SkyServe (/info)
  to trigger scale-up from zero.

Env vars:
  SKYSERVE_BASE_URL        e.g. http://44.201.119.3:30001
  CLOUDRUN_BASE_URL        e.g. https://your-sam-service-uc.a.run.app

  SKYSERVE_READY_PATH      default: /health   (traffic-driven readiness check)
  SKYSERVE_POKE_PATH       default: /info     (called per-request during cold start)

  PROBE_TIMEOUT_SECONDS    default: 1.0       (timeout for readiness check)
  CHECK_MIN_INTERVAL_SECONDS default: 1.0     (throttle readiness checks)

  POKE_TIMEOUT_SECONDS     default: 0.3       (keep tiny; do not block user traffic)
  UPSTREAM_TIMEOUT_SECONDS default: 120.0     (timeout for upstream requests)

  API_KEY                  if set, required for all requests
  API_KEY_HEADER           default: x-api-key (also accepts Authorization: Bearer)
  ALLOW_HEALTH_NO_AUTH     default: 0         (if "1", /router/health skips auth)

  ROUTE_WINDOW_SIZE        default: 200       (rolling window size for routing stats)

Run:
  pip install flask requests
  export API_KEY="your-static-key"
  export SKYSERVE_BASE_URL="http://<skyserve-endpoint>"
  export CLOUDRUN_BASE_URL="https://<cloudrun-url>"
  gunicorn -w 2 -k gthread --threads 8 --timeout 120 --bind 0.0.0.0:$PORT load_balancer:app
"""

from __future__ import annotations

import hmac
import logging
import os
import threading
import time
from typing import Dict
from collections import deque
from concurrent.futures import ThreadPoolExecutor


import requests
from flask import Flask, Response, request

app = Flask(__name__)
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
logger = logging.getLogger("router")

HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}


def env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    return default if v is None else float(v)


def env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y")


def join_url(base: str, path: str) -> str:
    base = base.rstrip("/")
    path = "/" + path.lstrip("/")
    return base + path

def filter_incoming_headers(h: Dict[str, str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    drop = {API_KEY_HEADER.lower(), "authorization"}
    for k, v in h.items():
        lk = k.lower()
        if lk in HOP_BY_HOP_HEADERS:
            continue
        if lk == "host":
            continue
        if lk in drop:
            continue
        out[k] = v
    return out


def filter_outgoing_headers(h: requests.structures.CaseInsensitiveDict) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k, v in h.items():
        lk = k.lower()
        if lk in HOP_BY_HOP_HEADERS:
            continue
        # Let Flask/Werkzeug manage content-length
        if lk == "content-length":
            continue
        out[k] = v
    return out


BG_MAX_WORKERS = int(os.getenv("BG_MAX_WORKERS", "4"))
EXECUTOR = ThreadPoolExecutor(max_workers=BG_MAX_WORKERS)

SKYSERVE_BASE_URL = os.environ.get("SKYSERVE_BASE_URL", "").rstrip("/")
CLOUDRUN_BASE_URL = os.environ.get("CLOUDRUN_BASE_URL", "").rstrip("/")

SKYSERVE_READY_PATH = os.getenv("SKYSERVE_READY_PATH", "/health")
SKYSERVE_POKE_PATH = os.getenv("SKYSERVE_POKE_PATH", "/info")

PROBE_TIMEOUT_SECONDS = env_float("PROBE_TIMEOUT_SECONDS", 1.0)
POKE_TIMEOUT_SECONDS = env_float("POKE_TIMEOUT_SECONDS", 0.3)
UPSTREAM_TIMEOUT_SECONDS = env_float("UPSTREAM_TIMEOUT_SECONDS", 210.0)

_last_check_ts = 0.0
CHECK_MIN_INTERVAL_SECONDS = env_float("CHECK_MIN_INTERVAL_SECONDS", 1.0)

API_KEY = os.getenv("API_KEY", "")
API_KEY_HEADER = os.getenv("API_KEY_HEADER", "x-api-key")
ALLOW_HEALTH_NO_AUTH = env_bool("ALLOW_HEALTH_NO_AUTH", False)

ROUTE_WINDOW_SIZE = int(os.getenv("ROUTE_WINDOW_SIZE", "200"))
_recent_routes = deque(maxlen=ROUTE_WINDOW_SIZE)

POKE_MIN_INTERVAL_SECONDS = env_float("POKE_MIN_INTERVAL_SECONDS", 0.5)
_last_poke_ts = 0.0

SESSION = requests.Session()

# Minimal shared state (simple + good enough to start)
_state_lock = threading.Lock()
_skyserve_ready = False
_last_probe_ts = None
_last_probe_err = None
_req_total = 0
_req_to_skyserve = 0
_req_to_cloudrun = 0

def record_route(chosen_backend: str) -> None:
    """chosen_backend: 'skyserve' or 'cloudrun'."""
    global _req_total, _req_to_skyserve, _req_to_cloudrun
    with _state_lock:
        _req_total += 1
        _recent_routes.append(chosen_backend)
        if chosen_backend == "skyserve":
            _req_to_skyserve += 1
        else:
            _req_to_cloudrun += 1

def route_stats() -> dict:
    with _state_lock:
        total = _req_total
        sky = _req_to_skyserve
        cr = _req_to_cloudrun
        recent = list(_recent_routes)
    recent_total = len(recent)
    recent_sky = sum(1 for r in recent if r == "skyserve")
    recent_cr = recent_total - recent_sky
    return {
        "total": total,
        "skyserve": sky,
        "cloudrun": cr,
        "pct_skyserve": (100.0 * sky / total) if total else 0.0,
        "pct_cloudrun": (100.0 * cr / total) if total else 0.0,
        "window_total": recent_total,
        "window_skyserve": recent_sky,
        "window_cloudrun": recent_cr,
        "window_pct_skyserve": (100.0 * recent_sky / recent_total) if recent_total else 0.0,
        "window_pct_cloudrun": (100.0 * recent_cr / recent_total) if recent_total else 0.0,
    }

def set_ready(val: bool, err: str | None = None) -> None:
    global _skyserve_ready, _last_probe_ts, _last_probe_err
    with _state_lock:
        _skyserve_ready = val
        _last_probe_ts = time.time()
        _last_probe_err = err


def is_ready() -> bool:
    with _state_lock:
        return _skyserve_ready


def snapshot_state() -> dict:
    with _state_lock:
        return {
            "skyserve_ready": _skyserve_ready,
            "last_probe_ts": _last_probe_ts,
            "last_probe_err": _last_probe_err,
            "skyserve_base_url": SKYSERVE_BASE_URL,
            "cloudrun_base_url": CLOUDRUN_BASE_URL,
            "ready_probe": join_url(SKYSERVE_BASE_URL, SKYSERVE_READY_PATH) if SKYSERVE_BASE_URL else None,
            "poke_url": join_url(SKYSERVE_BASE_URL, SKYSERVE_POKE_PATH) if SKYSERVE_BASE_URL else None,
        }


def extract_api_key(req) -> str:
    key = req.headers.get(API_KEY_HEADER)
    if not key:
        auth = req.headers.get("Authorization", "")
        if auth.lower().startswith("bearer "):
            key = auth.split(" ", 1)[1]
    return key or ""


def is_authorized(req) -> bool:
    if not API_KEY:
        return True
    provided = extract_api_key(req)
    if not provided:
        return False
    return hmac.compare_digest(provided, API_KEY)


def check_skyserve_ready_async() -> None:
    if not SKYSERVE_BASE_URL:
        return
    global _last_check_ts
    now = time.time()
    with _state_lock:
        if now - _last_check_ts < CHECK_MIN_INTERVAL_SECONDS:
            return
        _last_check_ts = now

    ready_url = join_url(SKYSERVE_BASE_URL, SKYSERVE_READY_PATH)
    def _do() -> None:
        try:
            r = requests.get(ready_url, timeout=PROBE_TIMEOUT_SECONDS)
            if 200 <= r.status_code < 300:
                set_ready(True, None)
            else:
                set_ready(False, f"status={r.status_code}")
        except Exception as e:
            set_ready(False, str(e))
    threading.Thread(target=_do, daemon=True).start()



def poke_skyserve_async() -> None:
    global _last_poke_ts

    now = time.time()
    with _state_lock:
        if now - _last_poke_ts < POKE_MIN_INTERVAL_SECONDS:
            return
        _last_poke_ts = now

    poke_url = join_url(SKYSERVE_BASE_URL, SKYSERVE_POKE_PATH)

    def _do() -> None:
        try:
            requests.get(poke_url, timeout=POKE_TIMEOUT_SECONDS)
        except Exception:
            pass

    EXECUTOR.submit(_do)


@app.route("/router/health", methods=["GET"])
def router_health() -> Response:
    if not ALLOW_HEALTH_NO_AUTH and not is_authorized(request):
        return Response("unauthorized", status=401)
    return Response(
        response=str(snapshot_state()),
        status=200,
        mimetype="text/plain",
    )


@app.route("/", defaults={"path": ""}, methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])
@app.route("/<path:path>", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])
def proxy(path: str) -> Response:
    if not CLOUDRUN_BASE_URL:
        return Response("CLOUDRUN_BASE_URL not set", status=500)
    if not is_authorized(request):
        return Response("unauthorized", status=401)

    # Decide backend
    if SKYSERVE_BASE_URL and is_ready():
        backend_base = SKYSERVE_BASE_URL
        record_route("skyserve")
    else:
        backend_base = CLOUDRUN_BASE_URL
        record_route("cloudrun")
        # During cold start, poke SkyServe on every request
        if SKYSERVE_BASE_URL:
            poke_skyserve_async()
            check_skyserve_ready_async()

    target_url = join_url(backend_base, path)
    s = route_stats()
    if s["total"] % 100 == 0:
        logger.info(
            "total=%s skyserve=%s (%.1f%%) cloudrun=%s (%.1f%%) ready=%s "
            "window=%s skyserve_window=%s (%.1f%%) cloudrun_window=%s (%.1f%%)",
            s["total"],
            s["skyserve"],
            s["pct_skyserve"],
            s["cloudrun"],
            s["pct_cloudrun"],
            is_ready(),
            s["window_total"],
            s["window_skyserve"],
            s["window_pct_skyserve"],
            s["window_cloudrun"],
            s["window_pct_cloudrun"],
        )
    # preserve query string
    if request.query_string:
        target_url = target_url + "?" + request.query_string.decode("utf-8")

    headers = filter_incoming_headers(dict(request.headers))
    data = request.get_data()  # bytes

    try:
        r = SESSION.request(
            method=request.method,
            url=target_url,
            headers=headers,
            data=data if data else None,
            allow_redirects=True,
            timeout=(2.0, UPSTREAM_TIMEOUT_SECONDS) # connect timeout 2 seconds, read timeout is UPSTREAM_TIMEOUT_SECONDS
        )
    except requests.RequestException as e:
        return Response(f"upstream_error: {e}", status=502)

    resp_headers = filter_outgoing_headers(r.headers)
    return Response(
        response=r.content,
        status=r.status_code,
        headers=resp_headers,
        mimetype=r.headers.get("content-type", None),
    )


if __name__ == "__main__":
    if not CLOUDRUN_BASE_URL:
        raise SystemExit("CLOUDRUN_BASE_URL is required")

    # Flask dev server (fine for a starter). For real use, run under gunicorn.
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8080")))