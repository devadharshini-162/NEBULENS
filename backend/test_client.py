"""
backend/test_client.py
======================
Lightweight, zero-dependency ASGI test client using Python's native asyncio and urllib.parse.
Enables fast, robust end-to-end endpoint testing without requiring httpx.
"""

import asyncio
import json
import urllib.parse
from typing import Any, Dict, Optional, Tuple
from backend.main import app


class ASGIResponse:
    def __init__(self, status_code: int, headers: Dict[str, str], body: bytes):
        self.status_code = status_code
        self.headers = headers
        self.body = body

    def json(self) -> Any:
        return json.loads(self.body.decode("utf-8"))

    @property
    def text(self) -> str:
        return self.body.decode("utf-8")


class ASGITestClient:
    def __init__(self, asgi_app=app):
        self.app = asgi_app

    def get(self, url: str, headers: Optional[Dict[str, str]] = None) -> ASGIResponse:
        return asyncio.run(self._request("GET", url, headers=headers))

    async def _request(
        self,
        method: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        body: bytes = b""
    ) -> ASGIResponse:
        parsed = urllib.parse.urlparse(url)
        path = parsed.path or "/"
        query_string = parsed.query.encode("utf-8")
        
        raw_headers = []
        if headers:
            for k, v in headers.items():
                raw_headers.append((k.lower().encode("utf-8"), v.encode("utf-8")))

        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": method.upper(),
            "path": path,
            "raw_path": path.encode("utf-8"),
            "query_string": query_string,
            "headers": raw_headers,
            "server": ("testserver", 80),
            "client": ("127.0.0.1", 12345),
        }

        response_status = 500
        response_headers = {}
        response_body = []

        async def receive():
            return {"type": "http.request", "body": body, "more_body": False}

        async def send(message):
            nonlocal response_status, response_headers, response_body
            if message["type"] == "http.response.start":
                response_status = message["status"]
                for k, v in message.get("headers", []):
                    response_headers[k.decode("utf-8")] = v.decode("utf-8")
            elif message["type"] == "http.response.body":
                response_body.append(message.get("body", b""))

        await self.app(scope, receive, send)
        return ASGIResponse(response_status, response_headers, b"".join(response_body))


test_client = ASGITestClient()
