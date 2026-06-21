# SPDX-License-Identifier: Apache-2.0
"""Pre-production HTTP receiver for Redis-backed parse jobs."""

from __future__ import annotations

import argparse
import json
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from pydantic import ValidationError

from tensorlake_docai.pipeline.api import ParseRequest
from tensorlake_docai.queue.job_store import FileJobStore
from tensorlake_docai.queue.models import JobRecord, QueueName
from tensorlake_docai.queue.redis_streams import RedisStreamsQueue, payload_from_raw


class ReceiverConfig:
    def __init__(self, redis_url: str, jobs_dir: str):
        self.redis_url = redis_url
        self.jobs_dir = jobs_dir


def make_handler(config: ReceiverConfig):
    store = FileJobStore(config.jobs_dir)
    queue = RedisStreamsQueue(config.redis_url)

    class ParseReceiver(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            if self.path != "/parse":
                self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
                return

            try:
                raw = self._read_json()
                request = ParseRequest.model_validate(raw)
            except (json.JSONDecodeError, ValidationError, ValueError) as e:
                self._json({"error": str(e)}, HTTPStatus.BAD_REQUEST)
                return

            record = JobRecord(request=request.model_dump(mode="json"))
            store.create(record)
            queue.enqueue(
                QueueName.CPU_INGEST,
                payload_from_raw(record.request, record.job_id),
            )
            self._json({"job_id": record.job_id, "status": record.status}, HTTPStatus.ACCEPTED)

        def do_GET(self) -> None:
            prefix = "/jobs/"
            if not self.path.startswith(prefix):
                self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
                return

            job_id = self.path[len(prefix) :]
            try:
                self._json(_redact_job_response(store.raw(job_id)), HTTPStatus.OK)
            except FileNotFoundError:
                self._json({"error": "job not found"}, HTTPStatus.NOT_FOUND)

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("content-length", "0"))
            if length <= 0:
                raise ValueError("empty request body")
            return json.loads(self.rfile.read(length))

        def _json(self, body: dict[str, Any], status: HTTPStatus) -> None:
            payload = json.dumps(body, default=str).encode()
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: Any) -> None:
            print(f"{self.address_string()} - {format % args}")

    return ParseReceiver


def _redact_job_response(record: dict[str, Any]) -> dict[str, Any]:
    redacted = dict(record)
    request = redacted.get("request")
    if isinstance(request, dict) and request.get("file_bytes"):
        request = dict(request)
        request["file_bytes"] = f"<redacted base64 bytes: {len(request['file_bytes'])} chars>"
        redacted["request"] = request
    return redacted


def serve(host: str, port: int, *, redis_url: str, jobs_dir: str) -> None:
    handler = make_handler(ReceiverConfig(redis_url, jobs_dir))
    server = ThreadingHTTPServer((host, port), handler)
    print(f"Listening on http://{host}:{port} with Redis at {redis_url}")
    server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--redis-url", default=os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    parser.add_argument("--jobs-dir", default=os.getenv("OPEN_INGEST_JOBS_DIR", "debug/jobs"))
    args = parser.parse_args()
    serve(args.host, args.port, redis_url=args.redis_url, jobs_dir=args.jobs_dir)


if __name__ == "__main__":
    main()
