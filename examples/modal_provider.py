# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env python
"""Modal-native provider endpoint backed by S3 state objects.

Required Modal secrets/env:
    PROVIDER_STATE_BUCKET
    AWS_REGION
    AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY, or the equivalent IAM role

Optional:
    PROVIDER_STATE_PREFIX
    PROVIDER_STATE_S3_ENDPOINT_URL
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import modal

APP_NAME = "stern-open-ingest-provider"
PROVIDER_SECRET_NAME = os.getenv("MODAL_PROVIDER_SECRET_NAME", "stern-open-ingest-provider-prod")
PARSE_MIN_CONTAINERS = int(os.getenv("MODAL_PROVIDER_PARSE_MIN_CONTAINERS", "1"))
GPU_MIN_CONTAINERS = int(os.getenv("MODAL_PROVIDER_GPU_MIN_CONTAINERS", "0"))
GPU_MAX_CONTAINERS = int(os.getenv("MODAL_PROVIDER_GPU_MAX_CONTAINERS", "10"))
REMOTE_ROOT = "/root/stern-open-ingest"
REPO_ROOT = Path(__file__).resolve().parents[1]

PADDLE_SERVER_URL = "http://127.0.0.1:8118/v1"
PADDLE_SERVER_MODELS_URL = f"{PADDLE_SERVER_URL}/models"
PADDLE_MODEL_NAME = os.getenv("PADDLE_OCR_VL_MODEL_NAME", "PaddleOCR-VL-1.6-0.9B")
PADDLE_VLLM_IMAGE = (
    "ccr-2vdh3abv-pub.cnc.bj.baidubce.com/"
    "paddlepaddle/paddleocr-genai-vllm-server:latest-nvidia-gpu"
)

base_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libmagic1", "poppler-utils")
    .pip_install(
        "tensorlake",
        "requests",
        "boto3",
        "fastapi[standard]",
        "pillow",
        "pillow-heif",
        "numpy",
        "pypdf",
        "pymupdf",
        "psutil",
        "img2pdf==0.6.3",
        "jdeskew==0.3.0",
        "pydantic",
        "python-magic==0.4.27",
        "markdownify",
    )
    .add_local_dir(
        REPO_ROOT,
        remote_path=REMOTE_ROOT,
        copy=True,
        ignore=[
            ".git",
            ".venv",
            ".venv-cursor",
            "__pycache__",
            ".pytest_cache",
            ".ruff_cache",
            "debug",
            "models",
            ".tensorlake",
            "benchmarks",
        ],
    )
    .run_commands(
        f"cd {REMOTE_ROOT} && python -m pip install -e . --no-deps --ignore-requires-python"
    )
)

gpu_image = (
    modal.Image.from_registry(PADDLE_VLLM_IMAGE)
    .entrypoint([])
    .apt_install("git", "libmagic1", "poppler-utils")
    .pip_install(
        "tensorlake",
        "requests",
        "boto3",
        "pillow-heif",
        "pypdf",
        "pymupdf",
        "img2pdf==0.6.3",
        "psutil",
        "jdeskew==0.3.0",
        "markdownify",
    )
    .run_commands(
        "python -m pip install paddlepaddle==3.2.1 "
        "-i https://www.paddlepaddle.org.cn/packages/stable/cpu/"
    )
    .run_commands("python -m pip install 'paddlex[ocr,genai-client]'")
    .add_local_dir(
        REPO_ROOT,
        remote_path=REMOTE_ROOT,
        copy=True,
        ignore=[
            ".git",
            ".venv",
            ".venv-cursor",
            "__pycache__",
            ".pytest_cache",
            ".ruff_cache",
            "debug",
            "models",
            ".tensorlake",
            "benchmarks",
        ],
    )
    .run_commands(
        f"cd {REMOTE_ROOT} && python -m pip install -e . --no-deps --ignore-requires-python"
    )
)

app = modal.App(APP_NAME)
provider_secret = modal.Secret.from_name(PROVIDER_SECRET_NAME)
_paddle_server: subprocess.Popen | None = None


def _store() -> Any:
    from tensorlake_docai.provider_state.storage import ProviderStateStore, S3ProviderStorage

    return ProviderStateStore(S3ProviderStorage.from_env())


def _wait_for_server(server: subprocess.Popen, timeout_seconds: int = 900) -> None:
    deadline = time.time() + timeout_seconds
    last_error: Exception | None = None
    while time.time() < deadline:
        exit_code = server.poll()
        if exit_code is not None:
            raise RuntimeError(
                "PaddleOCR-VL recognition server exited before becoming ready "
                f"(exit_code={exit_code})."
            )
        try:
            with urllib.request.urlopen(PADDLE_SERVER_MODELS_URL, timeout=5) as response:
                if response.status < 500:
                    return
        except (OSError, TimeoutError, urllib.error.URLError) as e:
            last_error = e
        time.sleep(5)
    raise RuntimeError(
        f"PaddleOCR-VL recognition server did not become ready at "
        f"{PADDLE_SERVER_MODELS_URL} within {timeout_seconds}s. Last error: {last_error}"
    )


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


def _ensure_paddle_server() -> None:
    global _paddle_server
    if _paddle_server is not None and _paddle_server.poll() is None:
        try:
            _wait_for_server(_paddle_server, timeout_seconds=10)
            return
        except RuntimeError:
            _paddle_server.terminate()
            try:
                _paddle_server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                _paddle_server.kill()

    _paddle_server = _start_paddle_server()
    _wait_for_server(_paddle_server)


@app.function(
    image=base_image,
    timeout=60,
    cpu=1,
    memory=1024,
    secrets=[provider_secret],
    min_containers=PARSE_MIN_CONTAINERS,
)
@modal.fastapi_endpoint(method="POST")
def parse(request: dict[str, Any]) -> dict[str, Any]:
    os.chdir(REMOTE_ROOT)
    sys.path.insert(0, f"{REMOTE_ROOT}/src")

    from tensorlake_docai.provider_state.orchestrator import create_provider_job

    state = create_provider_job(_store(), request)
    process_document.spawn(state.provider_job_id, request)
    return {"job_id": state.provider_job_id}


@app.function(image=base_image, timeout=60, cpu=1, memory=1024, secrets=[provider_secret])
@modal.fastapi_endpoint(method="GET")
def job_status(provider_job_id: str) -> dict[str, Any]:
    os.chdir(REMOTE_ROOT)
    sys.path.insert(0, f"{REMOTE_ROOT}/src")
    return _store().get_state(provider_job_id).redacted_dict()


@app.function(
    image=base_image,
    timeout=60 * 60,
    cpu=4,
    memory=8192,
    retries=modal.Retries(max_retries=2),
    secrets=[provider_secret],
)
def process_document(provider_job_id: str, request: dict[str, Any] | None = None) -> dict[str, Any]:
    os.chdir(REMOTE_ROOT)
    sys.path.insert(0, f"{REMOTE_ROOT}/src")

    from tensorlake_docai.provider_state.models import ProviderJobStatus
    from tensorlake_docai.provider_state.orchestrator import (
        mark_gpu_running,
        process_provider_job,
        run_pipeline_locally,
    )

    store = _store()

    def runner(raw_request: dict[str, Any], needs_gpu: bool) -> dict[str, Any]:
        if not needs_gpu:
            return run_pipeline_locally(raw_request, needs_gpu=False)
        mark_gpu_running(store, provider_job_id)
        return run_pipeline_on_gpu.remote(raw_request)

    state = process_provider_job(
        store,
        provider_job_id,
        raw_request=request,
        pipeline_runner=runner,
    )
    if state.status == ProviderJobStatus.WEBHOOK_RETRYING:
        retry_webhook.spawn(provider_job_id)
    return state.redacted_dict()


@app.function(
    image=gpu_image,
    gpu=os.getenv("MODAL_PROVIDER_GPU", "L4"),
    cpu=8,
    memory=32768,
    ephemeral_disk=524_288,
    timeout=60 * 60,
    retries=modal.Retries(max_retries=2),
    min_containers=GPU_MIN_CONTAINERS,
    max_containers=GPU_MAX_CONTAINERS,
)
def run_pipeline_on_gpu(raw_request: dict[str, Any]) -> dict[str, Any]:
    os.chdir(REMOTE_ROOT)
    sys.path.insert(0, f"{REMOTE_ROOT}/src")

    from tensorlake_docai.provider_state.orchestrator import run_pipeline_locally

    os.environ["ENABLE_GPU_OCR_TASKS"] = "1"
    os.environ["PADDLE_OCR_VL_SERVER_URL"] = PADDLE_SERVER_URL
    os.environ["PADDLE_OCR_VL_REC_BACKEND"] = "vllm-server"
    os.environ["PADDLE_OCR_VL_DEVICE"] = "cpu"

    _ensure_paddle_server()
    return run_pipeline_locally(raw_request, needs_gpu=True)


@app.function(
    image=base_image,
    timeout=30 * 60,
    retries=modal.Retries(max_retries=0),
    secrets=[provider_secret],
)
def retry_webhook(provider_job_id: str, max_attempts: int = 10) -> dict[str, Any]:
    os.chdir(REMOTE_ROOT)
    sys.path.insert(0, f"{REMOTE_ROOT}/src")

    from tensorlake_docai.provider_state.models import ProviderJobStatus
    from tensorlake_docai.provider_state.orchestrator import deliver_webhook_for_job

    store = _store()
    state = store.get_state(provider_job_id)

    while (
        state.status == ProviderJobStatus.WEBHOOK_RETRYING and state.webhook_attempts < max_attempts
    ):
        delay = min(300, 2 ** max(state.webhook_attempts, 1))
        time.sleep(delay)
        state = deliver_webhook_for_job(store, provider_job_id)

    if state.status == ProviderJobStatus.WEBHOOK_RETRYING:
        from tensorlake_docai.provider_state.orchestrator import fail_provider_job

        state = fail_provider_job(
            store,
            provider_job_id,
            f"Webhook delivery failed after {state.webhook_attempts} attempts",
        )
    return state.redacted_dict()
