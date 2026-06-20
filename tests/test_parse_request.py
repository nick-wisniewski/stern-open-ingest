# SPDX-License-Identifier: Apache-2.0
"""ParseRequest schema invariants — these are the public API surface."""

import pytest
from pydantic import ValidationError

from tensorlake_docai.pipeline.api import ParseRequest
from tensorlake_docai.pipeline.routing import _download_file


def test_minimal_request_with_bytes():
    req = ParseRequest(file_name="x.pdf", mime_type="application/pdf", file_bytes="aGVsbG8=")
    assert req.ocr_model == "dots-ocr"  # default
    assert req.table_output_mode == "markdown"


def test_minimal_request_with_url():
    req = ParseRequest(
        file_name="x.pdf",
        mime_type="application/pdf",
        file_url="https://bucket.s3.amazonaws.com/key?X-Amz-Signature=abc",
    )
    assert req.file_bytes is None


def test_direct_s3_url_download_rejected():
    with pytest.raises(Exception, match="HTTPS presigned URL"):
        _download_file("s3://bucket/key")


def test_unknown_ocr_model_rejected():
    with pytest.raises(ValidationError):
        ParseRequest(
            file_name="x.pdf",
            mime_type="application/pdf",
            file_bytes="aGVsbG8=",
            ocr_model="marker",  # dropped backend
        )


def test_paddle_ocr_vl_model_accepted():
    req = ParseRequest(
        file_name="x.pdf",
        mime_type="application/pdf",
        file_bytes="aGVsbG8=",
        ocr_model="paddle-ocr-vl",
    )
    assert req.ocr_model == "paddle-ocr-vl"


def test_legacy_model_codes_rejected():
    # Internal `model0X` names from the Inkwell codebase are no longer
    # accepted on the public API.
    for legacy in ("model01", "model02", "model03", "model05", "gemini3"):
        with pytest.raises(ValidationError):
            ParseRequest(
                file_name="x.pdf",
                mime_type="application/pdf",
                file_bytes="aGVsbG8=",
                ocr_model=legacy,
            )


def test_inkwell_only_field_removed():
    # `is_server_presigned_url` was Inkwell-specific and stripped on OSS publish.
    # If it sneaks back in, model_fields will pick it up.
    assert "is_server_presigned_url" not in ParseRequest.model_fields


def test_removed_enrichment_fields_stay_removed():
    removed_fields = {
        "chart_extraction",
        "detect_barcode",
        "figure_ocr_prompt",
        "figure_summarization",
        "figure_summarization_prompt",
        "include_full_page_image",
        "include_images",
        "chunk_strategy",
        "disable_layout_detection",
        "org_quota",
        "page_classification_request",
        "table_parsing_strategy",
        "table_summarization",
        "table_summarization_prompt",
    }
    assert removed_fields.isdisjoint(ParseRequest.model_fields)
