# SPDX-License-Identifier: Apache-2.0
"""Tests for final output serialization."""

from tensorlake_docai.pipeline.api import (
    Page,
    PageFragment,
    PageFragmentType,
    ParsedDocument,
    Text,
    Usage,
)
from tensorlake_docai.pipeline.output_formatter import _create_final_output


def test_create_final_output_omits_fragment_bboxes():
    parsed_document = ParsedDocument(
        chunks=[],
        pages=[
            Page(
                page_number=1,
                page_fragments=[
                    PageFragment(
                        fragment_type=PageFragmentType.TEXT,
                        content=Text(content="body"),
                        bbox={"x1": 1, "y1": 2, "x2": 3, "y2": 4},
                    )
                ],
            )
        ],
    )

    output = _create_final_output(parsed_document, Usage(pages_parsed=1))

    fragment = output["document"]["pages"][0]["page_fragments"][0]
    assert "bbox" not in fragment
