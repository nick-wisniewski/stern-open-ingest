# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env python
"""
Parse a single PDF or image with Open Ingest.

Usage:
    # Run the pipeline in-process
    python examples/parse_pdf.py --file my.pdf --local

    # With retained enrichment
    python examples/parse_pdf.py --file my.pdf --local \
        --table-merging --key-value-extraction --xpage-header-detection

Output: ./debug/document.json plus ./debug/document.md.
"""

import argparse
import base64
import urllib.error
import urllib.request
import shutil
import os
from pathlib import Path

from tensorlake.applications import run_local_application, run_remote_application

from tensorlake_docai.pipeline.api import (
    ParseRequest,
    ParsedDocument,
)
from tensorlake_docai.pipeline.file_converter import normalize_file_type_and_upload

PADDLE_OCR_MODEL = "paddle-ocr-vl"
PADDLE_SERVER_URL_ENV = "PADDLE_OCR_VL_SERVER_URL"
PADDLE_DEFAULT_SERVER_URL = "http://127.0.0.1:8118/v1"

MIME_BY_EXT = {
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".jpg": "image/jpg",
    ".jpeg": "image/jpeg",
    ".heif": "image/heif",
    ".heic": "image/heic",
}


def build_request(args: argparse.Namespace) -> ParseRequest:
    path = args.file

    if path.startswith("https://"):
        file_bytes = None
        file_url = path
        file_name = Path(path).name
        mime_type = MIME_BY_EXT.get(Path(path).suffix.lower(), "application/pdf")
    else:
        local = Path(path)
        if not local.exists():
            raise FileNotFoundError(path)
        file_bytes = base64.b64encode(local.read_bytes()).decode()
        file_url = None
        file_name = local.name
        mime_type = MIME_BY_EXT.get(local.suffix.lower(), "application/pdf")

    return ParseRequest(
        file_bytes=file_bytes,
        file_url=file_url,
        file_name=file_name,
        mime_type=mime_type,
        ocr_model=args.ocr_model,
        pages_to_parse=args.pages or None,
        table_output_mode=args.table_output_mode,
        table_merging=args.table_merging,
        key_value_extraction=args.key_value_extraction,
        xpage_header_detection=args.xpage_header_detection,
        ignore_sections=set(args.ignore_sections) if args.ignore_sections else None,
    )


def check_paddle_preflight(ocr_model: str, timeout: float = 3.0) -> None:
    if ocr_model != PADDLE_OCR_MODEL:
        return

    from tensorlake_docai.ocr.paddle_ocr_vl import _cuda_is_available

    if not _cuda_is_available():
        raise RuntimeError(
            "paddle-ocr-vl requires CUDA, but no usable GPU was detected. "
            "Run this smoke test on a CUDA-equipped host."
        )

    server_url = os.getenv(PADDLE_SERVER_URL_ENV, PADDLE_DEFAULT_SERVER_URL).rstrip("/")
    models_url = f"{server_url}/models"
    try:
        with urllib.request.urlopen(models_url, timeout=timeout) as response:
            if response.status >= 500:
                raise RuntimeError(
                    f"PaddleOCR-VL recognition server returned HTTP {response.status} "
                    f"from {models_url}."
                )
    except urllib.error.HTTPError as e:
        if e.code >= 500:
            raise RuntimeError(
                f"PaddleOCR-VL recognition server returned HTTP {e.code} from {models_url}."
            ) from e
    except (OSError, TimeoutError, urllib.error.URLError) as e:
        raise RuntimeError(
            f"PaddleOCR-VL recognition server is not reachable at {models_url}. "
            f"Start the local vLLM/SGLang server or set {PADDLE_SERVER_URL_ENV}."
        ) from e


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    core = parser.add_argument_group("core")
    core.add_argument("--file", required=True, help="Local path or presigned HTTPS URL")
    core.add_argument(
        "--ocr-model",
        default=PADDLE_OCR_MODEL,
        choices=["dots-ocr", PADDLE_OCR_MODEL],
        help="OCR backend (see docs/models.md). Default: paddle-ocr-vl.",
    )
    core.add_argument("--pages", type=int, nargs="*", help="Pages to parse (1-indexed)")
    core.add_argument(
        "--local", action="store_true", help="Run in-process instead of remote deploy"
    )
    core.add_argument(
        "--paddle-preflight",
        action="store_true",
        help=(
            "Check CUDA and the PaddleOCR-VL server before running. Use this when "
            "you already know the document should route to Paddle OCR."
        ),
    )
    core.add_argument("--out", default="debug", help="Output directory")

    output = parser.add_argument_group("output shape")
    output.add_argument(
        "--table-output-mode",
        default="markdown",
        choices=["markdown", "html"],
        help="Format of table content in the output",
    )
    output.add_argument(
        "--ignore-sections",
        nargs="*",
        default=[],
        help="PageFragmentType values to drop from output (e.g. page_footer figure)",
    )
    tables = parser.add_argument_group("table enrichment")
    tables.add_argument(
        "--table-merging",
        action="store_true",
        help="Stitch tables that span pages or are split by intervening content",
    )
    forms = parser.add_argument_group("forms / key-value")
    forms.add_argument(
        "--key-value-extraction",
        action="store_true",
        help="Extract key-value pairs from detected document regions",
    )

    xpage = parser.add_argument_group("cross-page heuristics")
    xpage.add_argument(
        "--xpage-header-detection",
        action="store_true",
        help="Detect repeating cross-page headers/footers",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    req = build_request(args)
    if args.paddle_preflight:
        check_paddle_preflight(req.ocr_model)

    runner = run_local_application if args.local else run_remote_application
    handle = runner(normalize_file_type_and_upload, req.model_dump())
    print(f"Request ID: {handle.id}")

    raw = handle.output()
    if not raw or "document" not in raw:
        raise RuntimeError("No document returned")

    parsed = ParsedDocument.model_validate(raw["document"])
    print(f"Parsed document with ocr_model={args.ocr_model}")

    out_dir = Path(args.out)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    (out_dir / "document.json").write_text(parsed.model_dump_json(indent=2))
    (out_dir / "document.md").write_text(parsed.document_markdown or "")

    print(f"Wrote results to {out_dir}/")


if __name__ == "__main__":
    main()
