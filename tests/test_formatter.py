# SPDX-License-Identifier: Apache-2.0
"""Tests for postprocess.formatter — markdown rendering of page fragments."""

import pytest

from tensorlake_docai.pipeline.api import (
    Figure,
    ListItem,
    PageFragment,
    PageFragmentType,
    ParseRequest,
    SectionHeader,
    Table,
    Text,
)
from tensorlake_docai.postprocess.formatter import (
    document_to_markdown,
    escape_header_content,
    escape_markdown_content,
    page_fragment_to_markdown,
    page_to_markdown,
)


@pytest.fixture
def request_md():
    return ParseRequest(
        file_name="x.pdf",
        mime_type="application/pdf",
        table_output_mode="markdown",
    )


@pytest.fixture
def request_html():
    return ParseRequest(
        file_name="x.pdf",
        mime_type="application/pdf",
        table_output_mode="html",
    )


# --- escape_markdown_content ---------------------------------------------


def test_escape_markdown_content_passthrough_for_empty():
    assert escape_markdown_content("") == ""
    assert escape_markdown_content(None) is None


def test_escape_markdown_content_escapes_leading_hash():
    out = escape_markdown_content("# heading-like")
    assert out.startswith("\\#")


def test_escape_markdown_content_leaves_plain_text():
    assert escape_markdown_content("nothing special") == "nothing special"


def test_escape_markdown_content_escapes_inline_hash():
    # The regex requires whitespace on BOTH sides of `#+` to escape — that's the
    # header-marker pattern. `word # other` triggers, `word #tag other` does not.
    out = escape_markdown_content("word # other")
    assert "\\#" in out

    untouched = escape_markdown_content("word #tag other")
    assert "\\#" not in untouched


def test_escape_markdown_content_preserves_newlines():
    text = "line one\nline two"
    assert "\n" in escape_markdown_content(text)


# --- escape_header_content ------------------------------------------------


def test_escape_header_content_escapes_all_hashes():
    assert escape_header_content("a # b ## c") == "a \\# b \\#\\# c"


def test_escape_header_content_empty():
    assert escape_header_content("") == ""
    assert escape_header_content(None) is None


# --- page_fragment_to_markdown ------------------------------------------


def _frag(fragment_type, content):
    return PageFragment(fragment_type=fragment_type, content=content)


def test_list_item_rendered_as_bullet(request_md):
    out = page_fragment_to_markdown(
        _frag(PageFragmentType.LIST_ITEM, ListItem(content="apples")), request_md
    )
    assert out == "* apples\n"


def test_section_header_uses_level_plus_one(request_md):
    # level 0 -> "#", level 1 -> "##", level 2 -> "###"
    for lvl, marker in [(0, "#"), (1, "##"), (2, "###")]:
        out = page_fragment_to_markdown(
            _frag(PageFragmentType.SECTION_HEADER, SectionHeader(content="Hello", level=lvl)),
            request_md,
        )
        assert out.strip().startswith(marker + " ")


def test_section_header_preserves_existing_markdown(request_md):
    out = page_fragment_to_markdown(
        _frag(PageFragmentType.SECTION_HEADER, SectionHeader(content="## already-md", level=1)),
        request_md,
    )
    # Already-markdown headers are returned without re-escaping.
    assert "## already-md" in out
    assert "\\#" not in out


def test_section_header_escapes_inline_hash(request_md):
    out = page_fragment_to_markdown(
        _frag(PageFragmentType.SECTION_HEADER, SectionHeader(content="Topic #1", level=1)),
        request_md,
    )
    # Trailing # in content should be escaped so it doesn't fight header markers.
    assert "\\#" in out


def test_text_fragment_appends_double_newline(request_md):
    out = page_fragment_to_markdown(
        _frag(PageFragmentType.TEXT, Text(content="plain body")), request_md
    )
    assert out.endswith("\n\n")
    assert "plain body" in out


@pytest.mark.parametrize(
    "ftype",
    [
        PageFragmentType.FORMULA,
        PageFragmentType.FORMULA_CAPTION,
        PageFragmentType.TABLE_CAPTION,
        PageFragmentType.FIGURE_CAPTION,
    ],
)
def test_caption_like_fragments_pass_through(ftype, request_md):
    out = page_fragment_to_markdown(_frag(ftype, Text(content="caption-x")), request_md)
    assert "caption-x" in out


def test_figure_renders_content(request_md):
    fig = Figure(content="raw figure text")
    out = page_fragment_to_markdown(_frag(PageFragmentType.FIGURE, fig), request_md)
    assert "### Figure" in out
    assert "raw figure text" in out


def test_chart_renders_as_figure_content(request_md):
    fig = Figure(content="chart OCR text")
    out = page_fragment_to_markdown(_frag(PageFragmentType.CHART, fig), request_md)
    assert "### Figure" in out
    assert "chart OCR text" in out


def test_table_uses_markdown_when_mode_is_markdown(request_md):
    table = Table(
        content="table",
        html="<table><tr><td>x</td></tr></table>",
        markdown="| x |\n|---|",
    )
    out = page_fragment_to_markdown(_frag(PageFragmentType.TABLE, table), request_md)
    assert "| x |" in out
    assert "<table>" not in out


def test_table_uses_html_when_mode_is_html(request_html):
    table = Table(content="table", html="<table><tr><td>x</td></tr></table>", markdown="| x |")
    out = page_fragment_to_markdown(_frag(PageFragmentType.TABLE, table), request_html)
    assert "<table>" in out
    assert "| x |" not in out


def test_table_summary_is_appended(request_md):
    table = Table(
        content="t",
        html="<table></table>",
        markdown="md",
        summary="key insight",
    )
    out = page_fragment_to_markdown(_frag(PageFragmentType.TABLE, table), request_md)
    assert "Table Summary" in out
    assert "key insight" in out


# --- document/page-level integration --------------------------------------


def test_document_to_markdown_concatenates_pages(request_md):
    from tensorlake_docai.pipeline.api import Page

    page1 = Page(
        page_number=1,
        page_fragments=[_frag(PageFragmentType.TEXT, Text(content="page1"))],
    )
    page2 = Page(
        page_number=2,
        page_fragments=[_frag(PageFragmentType.TEXT, Text(content="page2"))],
    )
    out = document_to_markdown([page1, page2], request_md)
    assert out.index("page1") < out.index("page2")


def test_page_to_markdown_joins_fragments(request_md):
    from tensorlake_docai.pipeline.api import Page

    page = Page(
        page_number=1,
        page_fragments=[
            _frag(PageFragmentType.SECTION_HEADER, SectionHeader(content="H", level=1)),
            _frag(PageFragmentType.TEXT, Text(content="body")),
        ],
    )
    out = page_to_markdown(page, request_md)
    assert "## H" in out
    assert "body" in out
