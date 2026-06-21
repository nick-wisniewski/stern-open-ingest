# SPDX-License-Identifier: Apache-2.0
"""Redis queue worker contracts for CPU and GPU pools."""

from __future__ import annotations

import argparse
import base64
import os
import socket
import subprocess
import time
import traceback
import urllib.error
import urllib.request
from typing import Any

import requests
from tensorlake.applications import run_local_application

from tensorlake_docai.pipeline.api import ParseRequest
from tensorlake_docai.pipeline.file_converter import (
    classify_pages_for_ocr,
    count_document_pages,
    normalize_file_type_and_upload,
    process_file_from_s3_or_url,
)
from tensorlake_docai.queue.job_store import FileJobStore
from tensorlake_docai.queue.models import JobStatus, QueueName
from tensorlake_docai.queue.redis_streams import RedisStreamsQueue, QueueMessage, payload_from_raw

PADDLE_SERVER_URL_ENV = "PADDLE_OCR_VL_SERVER_URL"
PADDLE_DEFAULT_SERVER_URL = "http://127.0.0.1:8118/v1"
PADDLE_MODEL_NAME = os.getenv("PADDLE_OCR_VL_MODEL_NAME", "PaddleOCR-VL-1.6-0.9B")


def run_worker(
    worker_type: str,
    *,
    redis_url: str,
    jobs_dir: str,
    once: bool = False,
    start_paddle_server: bool = False,
) -> None:
    paddle_server = None
    if worker_type == "gpu":
        os.environ["ENABLE_GPU_OCR_TASKS"] = "1"
        queues = [QueueName.GPU_OCR]
        if start_paddle_server:
            paddle_server = _start_paddle_server()
            _wait_for_paddle_server(paddle_server)
        _check_paddle_server()
    elif worker_type == "cpu":
        queues = [QueueName.CPU_INGEST, QueueName.CPU_POSTPROCESS]
    else:
        raise ValueError("worker_type must be 'cpu' or 'gpu'")

    store = FileJobStore(jobs_dir)
    queue = RedisStreamsQueue(redis_url)
    consumer = f"{worker_type}-{socket.gethostname()}-{os.getpid()}"
    print(f"Starting {worker_type} worker {consumer} for queues: {[q.value for q in queues]}")

    try:
        while True:
            message = queue.dequeue(queues, consumer=consumer)
            if message is None:
                if once:
                    return
                continue

            try:
                _process_message(message, queue, store)
                queue.ack(message)
            except Exception as e:
                error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
                print(error)
                try:
                    store.save_error(message.payload.job_id, error)
                except FileNotFoundError:
                    pass
                queue.retry_or_dead_letter(message, error=error)

            if once:
                return
    finally:
        if paddle_server is not None:
            paddle_server.terminate()
            try:
                paddle_server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                paddle_server.kill()


def _process_message(
    message: QueueMessage,
    queue: RedisStreamsQueue,
    store: FileJobStore,
) -> None:
    record = store.get(message.payload.job_id)
    record.mark(JobStatus.RUNNING, queue=message.queue)
    store.save(record)

    if message.queue == QueueName.CPU_INGEST:
        routed_request, target = _route_ingest_payload(message.payload.request)
        record.request = routed_request
        record.mark(JobStatus.QUEUED, queue=target)
        store.save(record)
        queue.enqueue(
            target, payload_from_raw(routed_request, record.job_id, message.payload.attempts)
        )
        return

    result = _run_pipeline(message.payload.request)
    record.result = result
    record.error = None
    record.mark(JobStatus.SUCCEEDED, queue=message.queue)
    store.save(record)
    _deliver_webhook(message.payload.request, record.job_id, result)


def _route_ingest_payload(raw: dict[str, Any]) -> tuple[dict[str, Any], QueueName]:
    request = ParseRequest.model_validate(raw)
    process_file_from_s3_or_url(request)
    total_pages = count_document_pages(request.file_bytes, request.mime_type, request.file_name)
    _validate_pages_to_parse(request, total_pages)
    classify_pages_for_ocr(request, total_pages)

    if isinstance(request.file_bytes, bytes):
        request.file_bytes = base64.b64encode(request.file_bytes).decode()
        request.file_url = None

    target = QueueName.GPU_OCR if request.ocr_pages else QueueName.CPU_POSTPROCESS
    return request.model_dump(mode="json"), target


def _validate_pages_to_parse(request: ParseRequest, total_pages: int) -> None:
    if not request.pages_to_parse:
        return

    valid_pages = [p for p in request.pages_to_parse if 1 <= p <= total_pages]
    invalid_pages = [p for p in request.pages_to_parse if p < 1 or p > total_pages]
    if invalid_pages and not valid_pages:
        first_invalid = sorted(invalid_pages)[0]
        raise ValueError(
            f"Invalid page range specified. Document has {total_pages} pages, "
            f"but page {first_invalid} was requested."
        )
    request.pages_to_parse = valid_pages


def _run_pipeline(raw: dict[str, Any]) -> dict[str, Any]:
    handle = run_local_application(normalize_file_type_and_upload, raw)
    result = handle.output()
    if not result or "document" not in result:
        raise RuntimeError(f"Pipeline returned no document: {result!r}")
    return result


def _deliver_webhook(raw: dict[str, Any], job_id: str, result: dict[str, Any]) -> None:
    webhook_url = raw.get("webhook_url")
    if not webhook_url:
        return

    payload = {"job_id": job_id, "status": "succeeded", "result": result}
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            response = requests.post(webhook_url, json=payload, timeout=(5, 30))
            response.raise_for_status()
            return
        except requests.RequestException as e:
            last_error = e
            time.sleep(attempt)

    raise RuntimeError(f"Webhook delivery failed for {job_id}: {last_error}")


def _check_paddle_server(timeout: float = 5.0) -> None:
    server_url = os.getenv(PADDLE_SERVER_URL_ENV, PADDLE_DEFAULT_SERVER_URL).rstrip("/")
    models_url = f"{server_url}/models"
    try:
        with urllib.request.urlopen(models_url, timeout=timeout) as response:
            if response.status >= 500:
                raise RuntimeError(f"PaddleOCR-VL server returned HTTP {response.status}")
    except urllib.error.HTTPError as e:
        if e.code >= 500:
            raise RuntimeError(f"PaddleOCR-VL server returned HTTP {e.code}") from e
    except (OSError, TimeoutError, urllib.error.URLError) as e:
        raise RuntimeError(
            f"PaddleOCR-VL recognition server is not reachable at {models_url}. "
            f"Start it beside the GPU worker or set {PADDLE_SERVER_URL_ENV}."
        ) from e


def _start_paddle_server() -> subprocess.Popen:
    command = [
        "paddleocr",
        "genai_server",
        "--model_name",
        PADDLE_MODEL_NAME,
        "--host",
        "0.0.0.0",
        "--port",
        "8118",
        "--backend",
        os.getenv("PADDLE_OCR_VL_SERVER_BACKEND", "vllm"),
    ]
    print("Starting PaddleOCR-VL recognition server:")
    print(" ".join(command))
    return subprocess.Popen(command)


def _wait_for_paddle_server(server: subprocess.Popen, timeout_seconds: int = 900) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        exit_code = server.poll()
        if exit_code is not None:
            raise RuntimeError(
                "PaddleOCR-VL recognition server exited before becoming ready "
                f"(exit_code={exit_code})."
            )
        try:
            _check_paddle_server(timeout=5.0)
            return
        except RuntimeError:
            time.sleep(5)
    raise RuntimeError("PaddleOCR-VL recognition server did not become ready in time.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("worker_type", choices=["cpu", "gpu"])
    parser.add_argument("--redis-url", default=os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    parser.add_argument("--jobs-dir", default=os.getenv("OPEN_INGEST_JOBS_DIR", "debug/jobs"))
    parser.add_argument("--once", action="store_true", help="Process one message and exit")
    parser.add_argument(
        "--start-paddle-server",
        action="store_true",
        help="Start a colocated PaddleOCR-VL recognition server for GPU workers.",
    )
    args = parser.parse_args()
    run_worker(
        args.worker_type,
        redis_url=args.redis_url,
        jobs_dir=args.jobs_dir,
        once=args.once,
        start_paddle_server=args.start_paddle_server,
    )


if __name__ == "__main__":
    main()
