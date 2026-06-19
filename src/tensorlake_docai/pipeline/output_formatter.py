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
from tensorlake_docai.extraction.chunking_functions import chunk_document


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
    """Create ParsedDocument from ParseResult using existing chunking logic."""
    parsed_document = chunk_document(result)
    parsed_document.total_pages = (
        result.document_layout.total_pages if result.document_layout else 0
    )

    if result.form_filling_result:
        parsed_document.filled_pdf_base64 = result.form_filling_result.filled_pdf_base64
        parsed_document.form_filling_metadata = result.form_filling_result.metadata

    return parsed_document


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
    output = ParsedDocumentRef(document=parsed_document.model_dump(), usage=usage).model_dump()
    return output
