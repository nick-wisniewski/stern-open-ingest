# SPDX-License-Identifier: Apache-2.0
"""Tests for final output serialization."""

from tensorlake_docai.pipeline.api import (
    ParsedDocument,
    Usage,
)
from tensorlake_docai.pipeline.output_formatter import _create_final_output


def test_create_final_output_returns_markdown_and_usage_only():
    parsed_document = ParsedDocument(document_markdown="body")

    output = _create_final_output(parsed_document, Usage(pages_parsed=1))

    assert output == {
        "document": {"document_markdown": "body"},
        "usage": {
            "pages_parsed": 1,
            "ocr_input_tokens_used": None,
            "ocr_output_tokens_used": None,
            "extraction_input_tokens_used": None,
            "extraction_output_tokens_used": None,
            "summarization_input_tokens_used": None,
            "summarization_output_tokens_used": None,
            "header_correction_input_tokens_used": None,
            "header_correction_output_tokens_used": None,
        },
    }
