# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env python
"""
Run a one-document PaddleOCR-VL smoke test on a Modal CUDA worker.

Usage:
    modal setup
    modal run examples/modal_paddle_smoke.py --file sample.pdf --pages 1

Output: ./debug/modal-paddle-smoke/document.json plus document.md.
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import modal

APP_NAME = "stern-open-ingest-paddle-smoke"
REMOTE_ROOT = "/root/stern-open-ingest"
PADDLE_SERVER_URL = "http://127.0.0.1:8118/v1"
PADDLE_SERVER_MODELS_URL = f"{PADDLE_SERVER_URL}/models"
PADDLE_MODEL_NAME = "PaddleOCR-VL-1.6-0.9B"
MODAL_GPU = "L4"
PADDLE_VLLM_IMAGE = (
    "ccr-2vdh3abv-pub.cnc.bj.baidubce.com/"
    "paddlepaddle/paddleocr-genai-vllm-server:latest-nvidia-gpu"
)

MIME_BY_EXT = {
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".jpg": "image/jpg",
    ".jpeg": "image/jpeg",
    ".heif": "image/heif",
    ".heic": "image/heic",
}

REPO_ROOT = Path(__file__).resolve().parents[1]

image = (
    modal.Image.from_registry(PADDLE_VLLM_IMAGE)
    .entrypoint([])
    .apt_install(
        "git",
        "libmagic1",
        "poppler-utils",
    )
    .pip_install(
        "python-magic==0.4.27",
        "pillow-heif",
        "pypdf",
        "pymupdf",
        "img2pdf==0.6.3",
        "psutil",
        "jdeskew==0.3.0",
        "markdownify",
        "requests",
    )
    .run_commands(
        "python -m pip install paddlepaddle==3.2.1 "
        "-i https://www.paddlepaddle.org.cn/packages/stable/cpu/"
    )
    .run_commands("python -m pip install 'paddlex[ocr,genai-client]'")
    .run_commands("python -m pip install tensorlake --no-deps")
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
        ],
    )
    .run_commands(
        f"cd {REMOTE_ROOT} && python -m pip install -e . --no-deps --ignore-requires-python"
    )
)

app = modal.App(APP_NAME)


def _parse_pages(pages: str) -> list[int]:
    return [int(page.strip()) for page in pages.split(",") if page.strip()]


def _mime_from_file_name(file_name: str) -> str:
    return MIME_BY_EXT.get(Path(file_name).suffix.lower(), "application/pdf")


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
        "vllm",
    ]
    print("Starting PaddleOCR-VL recognition server:")
    print(" ".join(command))
    return subprocess.Popen(command)


@app.function(
    image=image,
    gpu=MODAL_GPU,
    cpu=8,
    memory=32768,
    ephemeral_disk=524_288,
    timeout=60 * 60,
)
def run_smoke_on_gpu(
    file_bytes: bytes,
    file_name: str,
    pages: list[int],
    table_output_mode: str = "markdown",
) -> dict:
    os.chdir(REMOTE_ROOT)
    sys.path.insert(0, f"{REMOTE_ROOT}/src")
    os.environ["ENABLE_GPU_OCR_TASKS"] = "1"
    os.environ["PADDLE_OCR_VL_SERVER_URL"] = PADDLE_SERVER_URL
    os.environ["PADDLE_OCR_VL_REC_BACKEND"] = "vllm-server"
    os.environ["PADDLE_OCR_VL_DEVICE"] = "cpu"

    server = _start_paddle_server()
    try:
        _wait_for_server(server)

        from tensorlake.applications import run_local_application
        from tensorlake_docai.pipeline.api import ParseRequest, ParsedDocument
        from tensorlake_docai.pipeline.file_converter import normalize_file_type_and_upload

        request = ParseRequest(
            file_bytes=base64.b64encode(file_bytes).decode(),
            file_name=file_name,
            mime_type=_mime_from_file_name(file_name),
            ocr_model="paddle-ocr-vl",
            pages_to_parse=pages or None,
            table_output_mode=table_output_mode,
        )

        handle = run_local_application(normalize_file_type_and_upload, request.model_dump())
        raw = handle.output()
        if not raw or "document" not in raw:
            raise RuntimeError(f"No document returned from smoke run: {raw!r}")

        parsed = ParsedDocument.model_validate(raw["document"])
        return {
            "document": parsed.model_dump(mode="json"),
            "markdown": parsed.document_markdown or "",
        }
    finally:
        server.terminate()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()


@app.local_entrypoint()
def main(
    file: str,
    pages: str = "1",
    out: str = "debug/modal-paddle-smoke",
    table_output_mode: str = "markdown",
) -> None:
    input_path = Path(file)
    if not input_path.exists():
        raise FileNotFoundError(file)

    page_numbers = _parse_pages(pages)
    result = run_smoke_on_gpu.remote(
        input_path.read_bytes(),
        input_path.name,
        page_numbers,
        table_output_mode,
    )

    out_dir = Path(out)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    (out_dir / "document.json").write_text(json.dumps(result["document"], indent=2))
    (out_dir / "document.md").write_text(result["markdown"])

    print(f"Wrote Modal Paddle smoke results to {out_dir}/")
