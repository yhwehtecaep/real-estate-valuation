"""
healthcheck.py
===============
Tiny standalone script used by the Dockerfile's HEALTHCHECK directive.
Separate file instead of an inline shell one-liner so the port-from-env
logic doesn't have to survive multiple layers of shell/Dockerfile quoting.

Exits 0 (healthy) if GET /health responds; exits 1 otherwise.
"""
import os
import sys
import urllib.request

port = os.environ.get("PORT", "8000")
url = f"http://localhost:{port}/health"

try:
    with urllib.request.urlopen(url, timeout=3) as resp:
        sys.exit(0 if resp.status == 200 else 1)
except Exception:
    sys.exit(1)
