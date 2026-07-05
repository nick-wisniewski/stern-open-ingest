# SPDX-License-Identifier: Apache-2.0
"""Pipeline-level routing integration test.

Exercises the real routing predicates against constructed `ParseRequest` and
`ParseResult` objects — no network, no provider clients, no LLMs. The intent
is to lock in the public contract: for each supported `ocr_model`, the
file-convertor stage routes to the correct OCR provider task; once OCR
finishes, the post-OCR stage routes to output formatting or VLM extraction
based on the request flags + actual document content.

This is the highest-leverage pipeline test we can write without standing up
the full tensorlake workflow runtime.
"""

import pytest

from tensorlake_docai.models.intermediate_objects import ParseResult
from tensorlake_docai.models.layout_objects import (
    DocumentLayout,
    PageLayout,
    PageLayoutElement,
)
from tensorlake_docai.ocr import DEFAULT_OCR_MODEL, OCR_BACKENDS, resolve_ocr_backend
from tensorlake_docai.pipeline import routing
from tensorlake_docai.pipeline.api import PageFragmentType, ParseRequest

# ---- fixtures -------------------------------------------------------------


def _pdf_request(**overrides) -> ParseRequest:
    return ParseRequest(
        file_name="x.pdf",
        mime_type="application/pdf",
        file_bytes="aGVsbG8=",  # base64("hello")
        **overrides,
    )


def _element(fragment_type: PageFragmentType, ocr_text: str = "") -> PageLayoutElement:
    return PageLayoutElement(
        bbox=(0.0, 0.0, 100.0, 100.0),
        fragment_type=fragment_type,
        score=0.99,
        ocr_text=ocr_text,
    )


def _parse_result(request: ParseRequest, elements: list[PageLayoutElement]) -> ParseResult:
    page = PageLayout(elements=elements, shape=(100, 100), page_number=1)
    layout = DocumentLayout(pages=[page], scale_factor=1.0, total_pages=1)
    return ParseResult(document_layout=layout, request=request)


# ---- ocr_model → backend resolution --------------------------------------


# Verify that each public ocr_model resolves to a distinct backend class, and
# that the catch-all "non-text file needs OCR" predicate fires for every
# supported model. This is the same invariant the old per-backend predicates
# checked, but verified at the registry level.


@pytest.mark.parametrize("ocr_model", sorted(OCR_BACKENDS))
def test_each_ocr_model_resolves_to_a_distinct_backend(ocr_model):
    req = _pdf_request(ocr_model=ocr_model)
    assert routing.file_convertor_should_go_to_ocr(req)
    backend_cls = resolve_ocr_backend(ocr_model)
    # Each model must map to a unique class
    other_classes = {resolve_ocr_backend(m) for m in OCR_BACKENDS if m != ocr_model}
    assert backend_cls not in other_classes


def test_unknown_ocr_model_falls_back_to_default():
    """Unknown values should not crash — they fall back to DEFAULT_OCR_MODEL.
    pydantic's Literal validates real user input at the API boundary; this
    guards against internal callers passing through stale strings."""
    assert resolve_ocr_backend("model99") is resolve_ocr_backend(DEFAULT_OCR_MODEL)
    assert resolve_ocr_backend(None) is resolve_ocr_backend(DEFAULT_OCR_MODEL)


def test_pdf_file_goes_to_ocr_by_default():
    req = _pdf_request()
    assert not routing.file_convertor_should_go_to_output_formatter(req)
    assert routing.file_convertor_should_go_to_ocr(req)


# ---- post-OCR routing: bare parse → output formatter ---------------------


def test_post_ocr_bare_request_goes_to_output_formatter():
    req = _pdf_request()
    parse_result = _parse_result(req, [_element(PageFragmentType.TEXT, "hello")])

    assert routing.ocr_should_go_to_output_formatter(req)
    assert not routing.ocr_should_go_to_vlm_extraction(req, parse_result)


# ---- post-OCR routing: VLM tasks gated by document content ---------------


def test_key_value_extraction_skipped_when_no_candidate_regions_found():
    req = _pdf_request(ocr_model="dots-ocr", key_value_extraction=True)
    parse_result = _parse_result(req, [_element(PageFragmentType.TEXT, "no tables here")])

    # key_value_extraction=True but document has no candidate regions → don't go to VLM
    assert not routing.ocr_should_go_to_vlm_extraction(req, parse_result)
    assert routing.ocr_should_go_to_output_formatter(req) is False  # gate is set


def test_key_value_extraction_routes_to_vlm_when_forms_present():
    req = _pdf_request(ocr_model="dots-ocr", key_value_extraction=True)
    parse_result = _parse_result(req, [_element(PageFragmentType.FORM)])

    assert routing.ocr_should_go_to_vlm_extraction(req, parse_result)
