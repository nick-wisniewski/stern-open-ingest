# SPDX-License-Identifier: Apache-2.0
"""Tests for S3-backed provider state."""

from __future__ import annotations

import fitz
import pytest

from tensorlake_docai.provider_state.models import (
    ProviderJobState,
    ProviderJobStatus,
)
from tensorlake_docai.provider_state.orchestrator import (
    create_provider_job,
    deliver_webhook_for_job,
    prepare_pipeline_request,
    process_provider_job,
    store_provider_input,
)
from tensorlake_docai.provider_state.storage import InMemoryProviderStorage, ProviderStateStore


def _pdf_with_text(text: str) -> bytes:
    doc = fitz.open()
    page = doc.new_page(width=300, height=200)
    page.insert_text((30, 50), text)
    data = doc.tobytes()
    doc.close()
    return data


def _stored_job(pdf: bytes, **state_overrides) -> tuple[ProviderStateStore, ProviderJobState]:
    store = ProviderStateStore(InMemoryProviderStorage())
    state = ProviderJobState(
        external_job_id="rails-job-1",
        parse_request={
            "file_name": "policy.pdf",
            "mime_type": "application/pdf",
            "ocr_model": "paddle-ocr-vl",
            "table_output_mode": "markdown",
        },
        **state_overrides,
    )
    store.create_state(state)
    input_key = store.put_input(state.provider_job_id, ".pdf", pdf, content_type="application/pdf")
    state = store.update_state(
        state.provider_job_id,
        lambda current: current.transition(
            ProviderJobStatus.INPUT_STORED,
            input_object_key=input_key,
        ),
    )
    return store, state


def test_state_transition_rejects_invalid_jump():
    state = ProviderJobState(external_job_id="rails-job-1")

    with pytest.raises(ValueError):
        state.transition(ProviderJobStatus.WEBHOOK_DELIVERED)


def test_redacted_state_hides_webhook_url():
    state = ProviderJobState(
        external_job_id="rails-job-1",
        webhook_url="https://example.com/hook?signature=secret",
    )

    assert state.redacted_dict()["webhook_url"] == "<redacted>"


def test_create_provider_job_accepts_minimal_rails_request_without_downloading():
    store = ProviderStateStore(InMemoryProviderStorage())

    state = create_provider_job(
        store,
        {
            "file_url": "https://bucket.s3.amazonaws.com/path/policy.pdf?signature=secret",
            "webhook_url": "https://example.com/webhook",
        },
    )

    assert state.external_job_id == state.provider_job_id
    assert state.status == ProviderJobStatus.QUEUED
    assert state.parse_request["file_name"] == "policy.pdf"
    assert state.parse_request["mime_type"] == "application/pdf"
    assert state.parse_request["ocr_model"] == "paddle-ocr-vl"
    assert state.input_object_key is None


def test_store_provider_input_downloads_after_job_creation(monkeypatch):
    store = ProviderStateStore(InMemoryProviderStorage())
    pdf = _pdf_with_text("This policy page has enough extractable text to skip OCR.")
    raw_request = {
        "file_url": "https://bucket.s3.amazonaws.com/path/policy.pdf?signature=secret",
        "webhook_url": "https://example.com/webhook",
    }
    state = create_provider_job(store, raw_request)

    def download(request):
        request.file_bytes = pdf
        request.mime_type = "application/pdf"

    monkeypatch.setattr(
        "tensorlake_docai.provider_state.orchestrator.process_file_from_s3_or_url", download
    )

    state = store_provider_input(store, state.provider_job_id, raw_request)

    assert state.status == ProviderJobStatus.INPUT_STORED
    assert state.input_object_key == f"jobs/{state.provider_job_id}/input/original.pdf"


def test_prepare_pipeline_request_persists_cpu_classification():
    pdf = _pdf_with_text("This policy page has enough extractable text to skip OCR.")
    store, state = _stored_job(pdf)

    updated, raw_request, needs_gpu = prepare_pipeline_request(store, state.provider_job_id)

    assert not needs_gpu
    assert updated.status == ProviderJobStatus.CPU_EXTRACTING
    assert updated.ocr_pages == []
    assert updated.page_classification[0].route == "born_digital"
    assert raw_request["ocr_pages"] == []
    assert raw_request["file_url"] is None
    assert raw_request["file_bytes"]


def test_process_provider_job_stores_result_keys_without_inline_webhook():
    pdf = _pdf_with_text("This policy page has enough extractable text to skip OCR.")
    store, state = _stored_job(pdf)

    def runner(raw_request, needs_gpu):
        assert not needs_gpu
        assert raw_request["ocr_pages"] == []
        return {"document": {"document_markdown": "parsed markdown"}, "usage": {"pages_parsed": 1}}

    updated = process_provider_job(
        store,
        state.provider_job_id,
        pipeline_runner=runner,
        deliver_webhook=False,
    )

    assert updated.status == ProviderJobStatus.WEBHOOK_PENDING
    assert updated.result_object_key == f"jobs/{state.provider_job_id}/result/document.json"
    assert updated.markdown_object_key == f"jobs/{state.provider_job_id}/result/document.md"
    assert store.storage.get_bytes(updated.markdown_object_key) == b"parsed markdown"


def test_deliver_webhook_sends_job_ids_and_s3_keys(monkeypatch):
    pdf = _pdf_with_text("This policy page has enough extractable text to skip OCR.")
    store, state = _stored_job(pdf, webhook_url="https://example.com/webhook")
    store.update_state(
        state.provider_job_id,
        lambda current: current.transition(ProviderJobStatus.CLASSIFYING),
    )
    store.update_state(
        state.provider_job_id,
        lambda current: current.transition(ProviderJobStatus.CPU_EXTRACTING),
    )
    store.update_state(
        state.provider_job_id,
        lambda current: current.transition(ProviderJobStatus.POSTPROCESSING),
    )
    store.update_state(
        state.provider_job_id,
        lambda current: current.transition(
            ProviderJobStatus.RESULT_STORED,
            result_object_key=f"jobs/{state.provider_job_id}/result/document.json",
            markdown_object_key=f"jobs/{state.provider_job_id}/result/document.md",
        ),
    )
    store.update_state(
        state.provider_job_id,
        lambda current: current.transition(ProviderJobStatus.WEBHOOK_PENDING),
    )
    sent = {}

    class Response:
        def raise_for_status(self):
            return None

    def post(url, *, json, timeout):
        sent["url"] = url
        sent["json"] = json
        sent["timeout"] = timeout
        return Response()

    monkeypatch.setattr("requests.post", post)

    delivered = deliver_webhook_for_job(store, state.provider_job_id)

    assert delivered.status == ProviderJobStatus.WEBHOOK_DELIVERED
    assert sent["json"] == {
        "job_id": state.provider_job_id,
        "status": "succeeded",
        "result_object_key": f"jobs/{state.provider_job_id}/result/document.json",
        "markdown_object_key": f"jobs/{state.provider_job_id}/result/document.md",
    }
