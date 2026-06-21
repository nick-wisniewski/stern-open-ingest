# SPDX-License-Identifier: Apache-2.0
"""Tests for CPU born-digital routing and extraction."""

from __future__ import annotations

import fitz

from tensorlake_docai.models.intermediate_objects import ParseResult
from tensorlake_docai.models.layout_objects import DocumentLayout, PageLayout, PageLayoutElement
from tensorlake_docai.ocr.utils import BatchProcessor
from tensorlake_docai.pipeline.api import PageFragmentType, ParseRequest
from tensorlake_docai.pipeline.born_digital import (
    classify_pdf_pages,
    cpu_text_pages_for_request,
    extract_pdf_text_layer_pages,
    ocr_pages_from_classification,
)
from tensorlake_docai.pipeline.routing import file_convertor_should_go_to_ocr, route_after_ocr


def _pdf_with_text(page_texts: list[str]) -> bytes:
    doc = fitz.open()
    for text in page_texts:
        page = doc.new_page(width=300, height=200)
        if text:
            page.insert_text((30, 50), text)
    data = doc.tobytes()
    doc.close()
    return data


def test_classifier_marks_text_layer_pages_born_digital():
    pdf = _pdf_with_text(["This policy page has enough extractable text to skip OCR."])

    decisions = classify_pdf_pages(pdf, total_pages=1)

    assert decisions[0].route == "born_digital"
    assert ocr_pages_from_classification(decisions) == []


def test_classifier_marks_empty_pages_for_ocr():
    pdf = _pdf_with_text([""])

    decisions = classify_pdf_pages(pdf, total_pages=1)

    assert decisions[0].route == "needs_ocr"
    assert ocr_pages_from_classification(decisions) == [1]


def test_cpu_text_pages_exclude_ocr_pages():
    request = ParseRequest(
        file_name="x.pdf",
        mime_type="application/pdf",
        file_bytes="aGVsbG8=",
        pages_to_parse=[1, 2, 3],
        ocr_pages=[2],
    )

    assert cpu_text_pages_for_request(3, request) == [1, 3]


def test_extract_pdf_text_layer_pages_builds_page_layout():
    pdf = _pdf_with_text(["Named Insured: Example Company"])

    pages = extract_pdf_text_layer_pages(pdf, [1])

    assert len(pages) == 1
    assert pages[0].page_number == 1
    assert pages[0].elements[0].fragment_type == PageFragmentType.TEXT
    assert "Named Insured" in pages[0].elements[0].ocr_text


def test_empty_ocr_pages_means_skip_gpu_after_classification():
    request = ParseRequest(
        file_name="x.pdf",
        mime_type="application/pdf",
        file_bytes="aGVsbG8=",
        ocr_pages=[],
    )

    assert not file_convertor_should_go_to_ocr(request)


def test_route_after_ocr_sorts_mixed_pages_before_output():
    request = ParseRequest(
        file_name="x.pdf",
        mime_type="application/pdf",
        file_bytes="aGVsbG8=",
        ocr_pages=[2],
    )
    pages = [
        PageLayout(
            page_number=2,
            shape=(100, 100),
            elements=[
                PageLayoutElement(
                    bbox=(0, 0, 10, 10),
                    fragment_type=PageFragmentType.TEXT,
                    score=1,
                    ocr_text="second",
                )
            ],
        ),
        PageLayout(
            page_number=1,
            shape=(100, 100),
            elements=[
                PageLayoutElement(
                    bbox=(0, 0, 10, 10),
                    fragment_type=PageFragmentType.TEXT,
                    score=1,
                    ocr_text="first",
                )
            ],
        ),
    ]
    result = ParseResult(
        request=request,
        document_layout=DocumentLayout(pages=pages, scale_factor=1.0, total_pages=2),
    )

    output = route_after_ocr(result, log_prefix="test")

    assert output["document"]["document_markdown"].index("first") < output["document"][
        "document_markdown"
    ].index("second")


class _NoopBatchProcessor(BatchProcessor):
    async def process_batch(self, processing_batch, batch_number):
        return []


def test_batch_processor_uses_ocr_pages_for_rasterization():
    pdf = _pdf_with_text(["page one text", "page two text"])
    request = ParseRequest(
        file_name="x.pdf",
        mime_type="application/pdf",
        file_bytes="",
        pages_to_parse=[1, 2],
        ocr_pages=[2],
    )
    request.file_bytes = pdf
    result = ParseResult(
        request=request,
        document_layout=DocumentLayout(pages=[], scale_factor=1.0, total_pages=2),
    )

    batches = list(_NoopBatchProcessor().get_processing_batches(result))

    assert [batch.page_numbers for batch in batches] == [[2]]
