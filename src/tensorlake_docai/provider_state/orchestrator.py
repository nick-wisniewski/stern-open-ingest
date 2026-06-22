# SPDX-License-Identifier: Apache-2.0
"""S3-backed provider job orchestration."""

from __future__ import annotations

import base64
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import requests
from tensorlake.applications import RequestError as RequestException

from tensorlake_docai.pipeline.api import ParseRequest
from tensorlake_docai.pipeline.born_digital import (
    classify_pdf_pages,
    ocr_pages_from_classification,
)
from tensorlake_docai.pipeline.file_converter import (
    count_document_pages,
    process_file_from_s3_or_url,
)
from tensorlake_docai.provider_state.models import (
    ProviderJobState,
    ProviderJobStatus,
    ProviderPageClassification,
    ProviderParseRequest,
)
from tensorlake_docai.provider_state.storage import ProviderStateStore

PipelineRunner = Callable[[dict[str, Any], bool], dict[str, Any]]


def create_provider_job(
    store: ProviderStateStore,
    raw_request: dict[str, Any],
) -> ProviderJobState:
    provider_request = ProviderParseRequest.model_validate(raw_request)
    parse_request = _to_parse_request(provider_request)
    state = ProviderJobState(
        parse_request=_redacted_parse_request(parse_request),
        webhook_url=str(provider_request.webhook_url),
    )
    state.external_job_id = state.provider_job_id
    return store.create_state(state)


def store_provider_input(
    store: ProviderStateStore,
    provider_job_id: str,
    raw_request: dict[str, Any],
) -> ProviderJobState:
    state = store.get_state(provider_job_id)
    if state.input_object_key:
        return state

    provider_request = ProviderParseRequest.model_validate(raw_request)
    parse_request = _to_parse_request(provider_request)
    process_file_from_s3_or_url(parse_request)
    file_bytes = parse_request.file_bytes
    if not isinstance(file_bytes, bytes):
        raise RequestException(message="Provider input download did not return bytes")

    input_key = store.put_input(
        provider_job_id,
        Path(parse_request.file_name).suffix or ".bin",
        file_bytes,
        content_type=parse_request.mime_type,
    )
    return store.update_state(
        provider_job_id,
        lambda current: current.transition(
            ProviderJobStatus.INPUT_STORED,
            input_object_key=input_key,
            parse_request=_redacted_parse_request(parse_request),
            last_error=None,
        ),
    )


def process_provider_job(
    store: ProviderStateStore,
    provider_job_id: str,
    *,
    raw_request: dict[str, Any] | None = None,
    pipeline_runner: PipelineRunner | None = None,
    deliver_webhook: bool = True,
) -> ProviderJobState:
    try:
        state = store.get_state(provider_job_id)
        if not state.input_object_key:
            if raw_request is None:
                raise RuntimeError(f"Provider job {provider_job_id} has no stored input")
            store_provider_input(store, provider_job_id, raw_request)
        state, raw_pipeline_request, needs_gpu = prepare_pipeline_request(store, provider_job_id)
        runner = pipeline_runner or run_pipeline_locally
        result = runner(raw_pipeline_request, needs_gpu)
        if not result or "document" not in result:
            raise RuntimeError(f"Pipeline returned no document: {result!r}")

        store.update_state(
            provider_job_id,
            lambda current: current.transition(ProviderJobStatus.POSTPROCESSING),
        )
        result_key, markdown_key = store.put_result(provider_job_id, result)
        state = store.update_state(
            provider_job_id,
            lambda current: current.transition(
                ProviderJobStatus.RESULT_STORED,
                result_object_key=result_key,
                markdown_object_key=markdown_key,
                last_error=None,
            ),
        )
        state = store.update_state(
            provider_job_id,
            lambda current: current.transition(ProviderJobStatus.WEBHOOK_PENDING),
        )
        if deliver_webhook:
            state = deliver_webhook_for_job(store, provider_job_id)
        return state
    except Exception as e:
        error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        return fail_provider_job(store, provider_job_id, error)


def prepare_pipeline_request(
    store: ProviderStateStore,
    provider_job_id: str,
) -> tuple[ProviderJobState, dict[str, Any], bool]:
    state = store.update_state(
        provider_job_id,
        lambda current: current.transition(ProviderJobStatus.CLASSIFYING),
    )
    file_bytes = store.get_input(state)

    raw_pipeline_request = dict(state.parse_request)
    raw_pipeline_request["file_bytes"] = base64.b64encode(file_bytes).decode()
    raw_pipeline_request["file_url"] = None
    request = ParseRequest.model_validate(raw_pipeline_request)
    request.file_bytes = file_bytes

    total_pages = count_document_pages(file_bytes, request.mime_type, request.file_name)
    _validate_pages_to_parse(request, total_pages)
    classifications = _classify_request_pages(request, total_pages)
    raw_pipeline_request = request.model_dump(mode="json", exclude={"file_bytes"})
    raw_pipeline_request["file_bytes"] = base64.b64encode(file_bytes).decode()
    raw_pipeline_request["file_url"] = None

    needs_gpu = bool(request.ocr_pages)
    next_status = (
        ProviderJobStatus.GPU_OCR_QUEUED if needs_gpu else ProviderJobStatus.CPU_EXTRACTING
    )
    state = store.update_state(
        provider_job_id,
        lambda current: current.transition(
            next_status,
            total_pages=total_pages,
            ocr_pages=request.ocr_pages,
            page_classification=classifications,
            parse_request=_redacted_parse_request(request),
            last_error=None,
        ),
    )
    return state, raw_pipeline_request, needs_gpu


def mark_gpu_running(store: ProviderStateStore, provider_job_id: str) -> ProviderJobState:
    return store.update_state(
        provider_job_id,
        lambda current: current.transition(ProviderJobStatus.GPU_OCR_RUNNING),
    )


def run_pipeline_locally(raw_request: dict[str, Any], needs_gpu: bool = False) -> dict[str, Any]:
    if needs_gpu:
        mark = "GPU"
    else:
        mark = "CPU"
    print(f"Running provider pipeline locally on {mark} path")

    from tensorlake.applications import run_local_application

    from tensorlake_docai.pipeline.file_converter import normalize_file_type_and_upload

    handle = run_local_application(normalize_file_type_and_upload, raw_request)
    result = handle.output()
    if not isinstance(result, dict):
        raise RuntimeError(f"Pipeline returned non-dict result: {result!r}")
    return result


def deliver_webhook_for_job(
    store: ProviderStateStore,
    provider_job_id: str,
    *,
    timeout: tuple[float, float] = (5, 30),
) -> ProviderJobState:
    state = store.get_state(provider_job_id)
    if not state.webhook_url:
        return store.update_state(
            provider_job_id,
            lambda current: current.transition(ProviderJobStatus.WEBHOOK_DELIVERED),
        )

    payload = {
        "job_id": state.provider_job_id,
        "status": "succeeded",
        "result_object_key": state.result_object_key,
        "markdown_object_key": state.markdown_object_key,
    }
    try:
        response = requests.post(state.webhook_url, json=payload, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as e:
        error = f"{type(e).__name__}: {e}"
        return store.update_state(
            provider_job_id,
            lambda current: current.transition(
                ProviderJobStatus.WEBHOOK_RETRYING,
                webhook_attempts=current.webhook_attempts + 1,
                last_error=error,
            ),
        )

    return store.update_state(
        provider_job_id,
        lambda current: current.transition(
            ProviderJobStatus.WEBHOOK_DELIVERED,
            webhook_attempts=current.webhook_attempts + 1,
            last_error=None,
        ),
    )


def fail_provider_job(
    store: ProviderStateStore,
    provider_job_id: str,
    error: str,
) -> ProviderJobState:
    return store.update_state(
        provider_job_id,
        lambda current: current.transition(ProviderJobStatus.FAILED, last_error=error),
    )


def _to_parse_request(provider_request: ProviderParseRequest) -> ParseRequest:
    file_url = str(provider_request.file_url)
    file_name = _file_name_from_url(file_url)
    return ParseRequest(
        file_url=file_url,
        webhook_url=str(provider_request.webhook_url),
        file_name=file_name,
        mime_type=_mime_type_from_file_name(file_name),
        ocr_model=provider_request.ocr_model,
        table_output_mode="markdown",
    )


def _file_name_from_url(file_url: str) -> str:
    path = unquote(urlparse(file_url).path)
    file_name = Path(path).name
    return file_name or "document.pdf"


def _mime_type_from_file_name(file_name: str) -> str:
    suffix = Path(file_name).suffix.lower()
    return {
        ".pdf": "application/pdf",
        ".png": "image/png",
        ".jpg": "image/jpg",
        ".jpeg": "image/jpeg",
        ".heif": "image/heif",
        ".heic": "image/heic",
    }.get(suffix, "application/pdf")


def _redacted_parse_request(request: ParseRequest) -> dict[str, Any]:
    data = request.model_dump(mode="json", exclude={"file_bytes", "file_url", "webhook_url"})
    if data.get("ocr_pages") is None:
        data.pop("ocr_pages", None)
    return data


def _validate_pages_to_parse(request: ParseRequest, total_pages: int) -> None:
    if not request.pages_to_parse:
        return

    valid_pages = [p for p in request.pages_to_parse if 1 <= p <= total_pages]
    invalid_pages = [p for p in request.pages_to_parse if p < 1 or p > total_pages]
    if invalid_pages and not valid_pages:
        first_invalid = sorted(invalid_pages)[0]
        raise RequestException(
            message=(
                f"Invalid page range specified. Document has {total_pages} pages, "
                f"but page {first_invalid} was requested."
            )
        )
    request.pages_to_parse = valid_pages


def _classify_request_pages(
    request: ParseRequest,
    total_pages: int,
) -> list[ProviderPageClassification]:
    if request.ocr_pages is not None:
        request.ocr_pages = [
            page
            for page in sorted(set(request.ocr_pages))
            if 1 <= page <= total_pages
            and (not request.pages_to_parse or page in request.pages_to_parse)
        ]
        return [
            ProviderPageClassification(
                page_number=page,
                route="needs_ocr",
                reason="caller supplied ocr_pages",
            )
            for page in request.ocr_pages
        ]

    if request.mime_type != "application/pdf":
        request.ocr_pages = [1]
        return [
            ProviderPageClassification(
                page_number=1,
                route="needs_ocr",
                reason="image input",
            )
        ]

    decisions = classify_pdf_pages(
        request.file_bytes,
        total_pages=total_pages,
        pages_to_parse=request.pages_to_parse,
    )
    request.ocr_pages = ocr_pages_from_classification(decisions)
    return [
        ProviderPageClassification.model_validate(decision.model_dump()) for decision in decisions
    ]
