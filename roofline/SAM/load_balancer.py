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

Run:
  pip install flask requests
  export SKYSERVE_BASE_URL="http://<skyserve-endpoint>"
  export CLOUDRUN_BASE_URL="https://<cloudrun-url>"
  python router.py
"""

from __future__ import annotations

import os
import threading
import time
from typing import Dict

import requests
from flask import Flask, Response, request

app = Flask(__name__)

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


def join_url(base: str, path: str) -> str:
    base = base.rstrip("/")
    path = "/" + path.lstrip("/")
    return base + path


def filter_incoming_headers(h: Dict[str, str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k, v in h.items():
        lk = k.lower()
        if lk in HOP_BY_HOP_HEADERS:
            continue
        if lk == "host":
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


SKYSERVE_BASE_URL = os.environ.get("SKYSERVE_BASE_URL", "").rstrip("/")
CLOUDRUN_BASE_URL = os.environ.get("CLOUDRUN_BASE_URL", "").rstrip("/")

SKYSERVE_READY_PATH = os.getenv("SKYSERVE_READY_PATH", "/health")
SKYSERVE_POKE_PATH = os.getenv("SKYSERVE_POKE_PATH", "/info")

PROBE_TIMEOUT_SECONDS = env_float("PROBE_TIMEOUT_SECONDS", 1.0)
POKE_TIMEOUT_SECONDS = env_float("POKE_TIMEOUT_SECONDS", 0.3)

_last_check_ts = 0.0
CHECK_MIN_INTERVAL_SECONDS = env_float("CHECK_MIN_INTERVAL_SECONDS", 1.0)

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
        if chosen_backend == "skyserve":
            _req_to_skyserve += 1
        else:
            _req_to_cloudrun += 1

def route_stats() -> dict:
    with _state_lock:
        total = _req_total
        sky = _req_to_skyserve
        cr = _req_to_cloudrun
    return {
        "total": total,
        "skyserve": sky,
        "cloudrun": cr,
        "pct_skyserve": (100.0 * sky / total) if total else 0.0,
        "pct_cloudrun": (100.0 * cr / total) if total else 0.0,
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
    """Fire-and-forget poke that should never slow down user request."""
    if not SKYSERVE_BASE_URL:
        return

    poke_url = join_url(SKYSERVE_BASE_URL, SKYSERVE_POKE_PATH)

    def _do() -> None:
        try:
            # Tiny timeout: we only want to generate "traffic" to trigger scale-up.
            requests.get(poke_url, timeout=POKE_TIMEOUT_SECONDS)
        except Exception:
            # Intentionally ignored.
            pass

    threading.Thread(target=_do, daemon=True).start()


@app.route("/router/health", methods=["GET"])
def router_health() -> Response:
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
    if s["total"] % 10 == 0:
        print(f'[router] total={s["total"]} skyserve={s["skyserve"]} ({s["pct_skyserve"]:.1f}%) '
              f'cloudrun={s["cloudrun"]} ({s["pct_cloudrun"]:.1f}%) ready={is_ready()}')
    # preserve query string
    if request.query_string:
        target_url = target_url + "?" + request.query_string.decode("utf-8")

    headers = filter_incoming_headers(dict(request.headers))
    data = request.get_data()  # bytes

    try:
        r = requests.request(
            method=request.method,
            url=target_url,
            headers=headers,
            data=data if data else None,
            allow_redirects=True,
            timeout=None,  # keep simple; you can add timeouts later
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
    # Start background probe thread
    # t = threading.Thread(target=probe_loop, daemon=True)
    # t.start()

    # Flask dev server (fine for a starter). For real use, run under gunicorn.
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8080")))