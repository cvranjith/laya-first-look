"""Minimal client for the Jev (TypeSafe AI) System One API.

Reads the API key from the JEV_API_KEY environment variable. Get a key at
https://typesafe.ai and export it before running any test_jev_*.py script:

    export JEV_API_KEY=$(cat /path/to/your/jev-api-key)

Uses a persistent Session (connection pooling / keep-alive) rather than a
fresh connection per call -- an early version of this client opened a new
TCP+TLS connection on every request, which added ~600ms of pure connection
setup on top of the ~80ms the server actually spends, and made a naive
sequential benchmark look far slower than the API really is.
"""
import os
import time

import requests
from requests.adapters import HTTPAdapter

ENDPOINT = "https://api.typesafe.ai/v1/systemone"

_session = requests.Session()
_session.mount("https://", HTTPAdapter(pool_connections=20, pool_maxsize=20))


def predict(state, questions, model="jev-latest"):
    """Returns (response_json, client_elapsed_seconds, server_time_ms).

    client_elapsed is the full round trip this process observed (network +
    server). server_time_ms comes from the `x-envoy-upstream-service-time`
    response header -- TypeSafe's gateway reporting how long its own upstream
    inference service took, which is almost certainly what their playground
    displays as "response time". It's None if the header is absent.
    """
    api_key = os.environ.get("JEV_API_KEY")
    if not api_key:
        raise RuntimeError("Set JEV_API_KEY before running (see module docstring).")

    t0 = time.perf_counter()
    resp = _session.post(
        ENDPOINT,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={"state": state, "model": model, "questions": questions},
        timeout=30,
    )
    elapsed = time.perf_counter() - t0
    resp.raise_for_status()
    server_ms = resp.headers.get("x-envoy-upstream-service-time")
    return resp.json(), elapsed, (float(server_ms) if server_ms is not None else None)
