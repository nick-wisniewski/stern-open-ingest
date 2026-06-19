# SPDX-License-Identifier: Apache-2.0
"""Tests for pure-logic routing predicates and helpers in pipeline/routing.py."""

import pytest

from tensorlake_docai.models.intermediate_objects import ParseResult
from tensorlake_docai.models.layout_objects import DocumentLayout, PageLayout, PageLayoutElement
from tensorlake_docai.pipeline.api import (
    PageFragmentType,
    ParseRequest,
)
from tensorlake_docai.pipeline.routing import (
    _check_has_table_and_figure_and_chart_and_form,
    dots_ocr_should_go_to_output_formatter,
    dots_ocr_should_go_to_vlm_extraction,
    file_convertor_should_go_to_ocr,
    file_convertor_should_go_to_output_formatter,
    file_convertor_should_go_to_vlm_extraction,
    handle_processing_error,
    is_markdown_table,
    markdown_to_html_table,
    ocr_should_go_to_output_formatter,
    ocr_should_go_to_vlm_extraction,
    should_route_to_table_merging,
    vlm_extraction_should_go_to_output_formatter,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _req(**kwargs) -> ParseRequest:
    defaults = dict(file_name="test.pdf", mime_type="application/pdf")
    defaults.update(kwargs)
    return ParseRequest(**defaults)


def _layout_with(*fragment_types) -> DocumentLayout:
    elements = [
        PageLayoutElement(
            bbox=(0.0, 0.0, 1.0, 1.0),
            fragment_type=ft,
            score=1.0,
        )
        for ft in fragment_types
    ]
    page = PageLayout(elements=elements, shape=(100, 100), page_number=1)
    return DocumentLayout(pages=[page], scale_factor=1.0, total_pages=1)


def _parse_result(request: ParseRequest, layout: DocumentLayout = None) -> ParseResult:
    if layout is None:
        layout = DocumentLayout(pages=[], scale_factor=1.0, total_pages=0)
    return ParseResult(request=request, document_layout=layout)


# ---------------------------------------------------------------------------
# is_markdown_table
# ---------------------------------------------------------------------------


def test_is_markdown_table_standard():
    md = "| A | B |\n|---|---|\n| 1 | 2 |"
    assert is_markdown_table(md)


def test_is_markdown_table_false_for_plain_text():
    assert not is_markdown_table("Just some text")


def test_is_markdown_table_false_for_empty():
    assert not is_markdown_table("")
    assert not is_markdown_table(None)


def test_is_markdown_table_single_line():
    assert not is_markdown_table("| A |")


# ---------------------------------------------------------------------------
# markdown_to_html_table
# ---------------------------------------------------------------------------


def test_markdown_to_html_table_produces_table_tag():
    md = "| Name | Age |\n|------|-----|\n| Alice | 30 |"
    html = markdown_to_html_table(md)
    assert "<table>" in html
    assert "<thead>" in html
    assert "<tbody>" in html
    assert "<th>Name</th>" in html
    assert "<td>Alice</td>" in html


def test_markdown_to_html_table_returns_input_when_no_separator():
    plain = "no separator here"
    assert markdown_to_html_table(plain) == plain


def test_markdown_to_html_table_with_pre_table_text():
    md = "Caption text\n| A | B |\n|---|---|\n| 1 | 2 |"
    html = markdown_to_html_table(md)
    assert "<div>Caption text</div>" in html
    assert "<table>" in html


# ---------------------------------------------------------------------------
# handle_processing_error
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "message,expected_fragment",
    [
        ("rate limit exceeded", "busy"),
        ("quota exceeded", "usage limit"),
        ("timed out processing", "timed out"),
        ("credentials invalid", "authentication"),
        ("corrupted file invalid pdf", "corrupted"),
        ("out of memory", "large"),
        ("unexpected failure", "processing failed"),
    ],
)
def test_handle_processing_error_categorisation(message, expected_fragment):
    result = handle_processing_error(Exception(message), "test context")
    assert expected_fragment.lower() in result.lower()


# ---------------------------------------------------------------------------
# _check_has_table_and_figure_and_chart_and_form
# ---------------------------------------------------------------------------


def test_check_has_table_detects_table():
    layout = _layout_with(PageFragmentType.TABLE)
    result = _parse_result(_req(), layout)
    has_table, has_figure, has_chart, has_form = _check_has_table_and_figure_and_chart_and_form(
        result
    )
    assert has_table
    assert not has_figure


def test_check_has_figure_detects_figure():
    layout = _layout_with(PageFragmentType.FIGURE)
    result = _parse_result(_req(), layout)
    _, has_figure, _, _ = _check_has_table_and_figure_and_chart_and_form(result)
    assert has_figure


def test_check_has_chart_detects_chart():
    layout = _layout_with(PageFragmentType.CHART)
    result = _parse_result(_req(), layout)
    _, _, has_chart, _ = _check_has_table_and_figure_and_chart_and_form(result)
    assert has_chart


def test_check_has_form_detects_form():
    layout = _layout_with(PageFragmentType.FORM)
    result = _parse_result(_req(), layout)
    _, _, _, has_form = _check_has_table_and_figure_and_chart_and_form(result)
    assert has_form


def test_check_all_false_for_empty_layout():
    result = _parse_result(_req())
    assert _check_has_table_and_figure_and_chart_and_form(result) == (False, False, False, False)


def test_check_detects_markdown_table_in_text_element():
    md_table = "| A | B |\n|---|---|\n| 1 | 2 |"
    elements = [
        PageLayoutElement(
            bbox=(0.0, 0.0, 1.0, 1.0),
            fragment_type=PageFragmentType.TEXT,
            score=1.0,
            ocr_text=md_table,
        )
    ]
    page = PageLayout(elements=elements, shape=(100, 100), page_number=1)
    layout = DocumentLayout(pages=[page], scale_factor=1.0, total_pages=1)
    result = _parse_result(_req(), layout)
    has_table, _, _, _ = _check_has_table_and_figure_and_chart_and_form(result)
    assert has_table


# ---------------------------------------------------------------------------
# file_convertor_should_go_to_* predicates
# ---------------------------------------------------------------------------


def test_file_convertor_pdf_goes_to_ocr():
    req = _req(mime_type="application/pdf")
    assert file_convertor_should_go_to_ocr(req)
    assert not file_convertor_should_go_to_output_formatter(req)


def test_file_convertor_never_goes_directly_to_vlm():
    req = _req(mime_type="application/pdf", key_value_extraction=True)
    assert not file_convertor_should_go_to_vlm_extraction(req)
    assert file_convertor_should_go_to_ocr(req)


# ---------------------------------------------------------------------------
# ocr_should_go_to_* predicates
# ---------------------------------------------------------------------------


def test_ocr_should_go_to_output_formatter_when_no_extras():
    req = _req()
    assert ocr_should_go_to_output_formatter(req)


def test_ocr_should_go_to_vlm_when_key_value_region_present():
    req = _req(key_value_extraction=True)
    layout = _layout_with(PageFragmentType.FORM)
    result = _parse_result(req, layout)
    assert ocr_should_go_to_vlm_extraction(req, result)


def test_ocr_should_go_to_vlm_false_when_no_key_value_candidates():
    req = _req(key_value_extraction=True)
    result = _parse_result(req)
    assert not ocr_should_go_to_vlm_extraction(req, result)


# ---------------------------------------------------------------------------
# should_route_to_table_merging
# ---------------------------------------------------------------------------


def test_should_route_to_table_merging_true():
    req = _req(table_merging=True)
    layout = _layout_with(PageFragmentType.TABLE)
    result = _parse_result(req, layout)
    assert should_route_to_table_merging(req, result)


def test_should_route_to_table_merging_false_no_table():
    req = _req(table_merging=True)
    result = _parse_result(req)
    assert not should_route_to_table_merging(req, result)


def test_should_route_to_table_merging_false_flag_off():
    req = _req(table_merging=False)
    layout = _layout_with(PageFragmentType.TABLE)
    result = _parse_result(req, layout)
    assert not should_route_to_table_merging(req, result)


# ---------------------------------------------------------------------------
# vlm_extraction_should_go_to_* predicates
# ---------------------------------------------------------------------------


def test_vlm_should_go_to_output_formatter_when_no_se():
    assert vlm_extraction_should_go_to_output_formatter(_req())


# ---------------------------------------------------------------------------
# dots_ocr_should_go_to_* predicates
# ---------------------------------------------------------------------------


def test_dots_ocr_output_formatter_when_nothing_needed():
    req = _req()
    result = _parse_result(req)
    assert dots_ocr_should_go_to_output_formatter(req, result)


def test_dots_ocr_vlm_when_key_value_candidate_present():
    req = _req(key_value_extraction=True)
    layout = _layout_with(PageFragmentType.FORM)
    result = _parse_result(req, layout)
    assert dots_ocr_should_go_to_vlm_extraction(req, result)
