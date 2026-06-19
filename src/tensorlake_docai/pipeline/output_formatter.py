# SPDX-License-Identifier: Apache-2.0
"""
Output Formatter

This module provides a function to format a ParseResult from previous steps into
the final ParsedDocumentRef output format, handling all token aggregation
consistently.
"""

from typing import Optional

from tensorlake_docai.pipeline.api import (
    ParsedDocumentRef,
    ParsedDocument,
    Usage,
)
from tensorlake_docai.models.intermediate_objects import ParseResult
from tensorlake_docai.postprocess.formatter import (
    document_layout_to_document,
    document_to_markdown,
)


def format_final_output(
    result: ParseResult,
) -> Optional[dict]:
    """
    Format the final output from any pipeline stage.

    Args:
        result: The ParseResult containing document layout and request info

    Returns:
        Dict representation of ParsedDocumentRef
    """
    print("=== format_final_output ===")
    parsed_document = _create_parsed_document(result)
    usage = _calculate_usage(result)
    final_output = _create_final_output(parsed_document, usage)
    print("=== end of format_final_output ===")

    return final_output


def _create_parsed_document(result: ParseResult) -> ParsedDocument:
    """Create the Rails-facing ParsedDocument from ParseResult."""
    if not result.document_layout:
        return ParsedDocument(document_markdown="")

    pages = document_layout_to_document(
        result.document_layout.pages,
        result.document_layout.scale_factor,
        result.request.ignore_sections,
        merged_tables=result.document_layout.merged_tables,
    )
    return ParsedDocument(document_markdown=document_to_markdown(pages, result.request))


def _calculate_usage(
    result: ParseResult,
) -> Usage:
    """
    Calculate total usage metrics by aggregating tokens from VLM and LLM tasks.
    """
    u = result.usage

    # Calculate pages parsed
    pages_parsed = 0
    if result.document_layout and result.document_layout.pages:
        pages_parsed = len(result.document_layout.pages)
    elif u and u.pages_parsed:
        pages_parsed = u.pages_parsed

    return Usage(
        pages_parsed=pages_parsed,
        ocr_input_tokens_used=(u.ocr_input_tokens_used or 0) if u else 0,
        ocr_output_tokens_used=(u.ocr_output_tokens_used or 0) if u else 0,
        extraction_input_tokens_used=(u.extraction_input_tokens_used or 0) if u else 0,
        extraction_output_tokens_used=(u.extraction_output_tokens_used or 0) if u else 0,
        summarization_input_tokens_used=(u.summarization_input_tokens_used or 0) if u else 0,
        summarization_output_tokens_used=(u.summarization_output_tokens_used or 0) if u else 0,
        header_correction_input_tokens_used=(
            (u.header_correction_input_tokens_used or 0) if u else 0
        ),
        header_correction_output_tokens_used=(
            (u.header_correction_output_tokens_used or 0) if u else 0
        ),
    )


def _create_final_output(parsed_document: ParsedDocument, usage: Usage) -> Optional[dict]:
    """Create the final ParsedDocumentRef output."""
    document = {"document_markdown": parsed_document.document_markdown}
    output = ParsedDocumentRef(document=document, usage=usage).model_dump()
    return output
