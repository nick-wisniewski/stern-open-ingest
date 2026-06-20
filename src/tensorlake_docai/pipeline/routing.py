# SPDX-License-Identifier: Apache-2.0
"""
Module with utility functions to use by workflow steps.
"""

import asyncio
import base64
import re
import time
from typing import Tuple

import requests
from tensorlake_docai.pipeline.api import PageFragmentType, ParseRequest
from tensorlake_docai.models.intermediate_objects import FileData, ParseResult
from tensorlake.applications import RequestError as RequestException

# File type mapping - shared with file_convertor.py
FILE_TYPE_MAPPING = {
    "application/pdf": "pdf",
    "image/png": "png",
    "image/jpeg": "jpeg",
    "image/jpg": "jpg",
    "image/heif": "heif",
    "image/heic": "heic",
}


def is_markdown_table(text: str) -> bool:
    if not text:
        return False
    lines = text.strip().split("\n")
    if len(lines) < 2:
        return False
    # Check for separator line with dashes and pipes
    for line in lines:
        if re.match(r"^\s*\|?[\s\-:|]+\|?\s*$", line) and "-" in line and "|" in line:
            return True

    # Check for ! separator tables
    # Heuristic: multiple lines with multiple !
    bang_lines_count = 0
    non_empty_lines = [line for line in lines if line.strip()]
    for line in non_empty_lines:
        if line.count("!") >= 2:
            bang_lines_count += 1

    if (
        len(non_empty_lines) > 0
        and bang_lines_count >= 2
        and bang_lines_count >= len(non_empty_lines) * 0.5
    ):
        return True
    return False


def markdown_to_html_table(markdown: str) -> str:
    all_lines = markdown.strip().split("\n")

    # Identify non-empty lines for structure detection
    non_empty_indices = [i for i, line in enumerate(all_lines) if line.strip()]

    if len(non_empty_indices) < 2:
        return markdown

    # Find separator index in non_empty_indices
    separator_map_idx = -1
    separator_char = "|"

    for idx in non_empty_indices:
        line = all_lines[idx]
        if re.match(r"^\s*\|?[\s\-:|]+\|?\s*$", line) and "-" in line and "|" in line:
            separator_map_idx = idx
            separator_char = "|"
            break

    # Check for ! table if no standard separator found
    if separator_map_idx == -1:
        bang_lines_count = 0
        for idx in non_empty_indices:
            if all_lines[idx].count("!") >= 2:
                bang_lines_count += 1

        if bang_lines_count >= 2 and bang_lines_count >= len(non_empty_indices) * 0.5:
            separator_char = "!"
        else:
            return markdown

    html_parts = []

    if separator_map_idx != -1:
        # Find header (previous non-empty line)
        try:
            sep_pos = non_empty_indices.index(separator_map_idx)
        except ValueError:
            return markdown

        if sep_pos == 0:
            return markdown

        header_map_idx = non_empty_indices[sep_pos - 1]

        # Pre-table text
        if header_map_idx > 0:
            pre_text = "\n".join(all_lines[:header_map_idx]).strip()
            if pre_text:
                html_parts.append(f"<div>{pre_text}</div>")

        html_parts.append("<table>")

        # Header
        header_line = all_lines[header_map_idx]
        header_content = header_line.strip()
        if header_content.startswith(separator_char):
            header_content = header_content[1:]
        if header_content.endswith(separator_char):
            header_content = header_content[:-1]

        header_cells = [cell.strip() for cell in header_content.split(separator_char)]
        html_parts.append("<thead><tr>")
        for cell in header_cells:
            html_parts.append(f"<th>{cell}</th>")
        html_parts.append("</tr></thead>")

        # Body
        html_parts.append("<tbody>")
        for idx in non_empty_indices[sep_pos + 1 :]:
            line = all_lines[idx]
            row_content = line.strip()
            if separator_char in row_content:
                if row_content.startswith(separator_char):
                    row_content = row_content[1:]
                if row_content.endswith(separator_char):
                    row_content = row_content[:-1]

                cells = [cell.strip() for cell in row_content.split(separator_char)]
                html_parts.append("<tr>")
                for cell in cells:
                    html_parts.append(f"<td>{cell}</td>")
                html_parts.append("</tr>")
        html_parts.append("</tbody>")
        html_parts.append("</table>")

    else:
        # Table without separator line (likely ! table)
        # Treat all lines as body
        html_parts.append("<table>")
        html_parts.append("<tbody>")

        for idx in non_empty_indices:
            line = all_lines[idx]
            row_content = line.strip()

            if separator_char in row_content:
                if row_content.startswith(separator_char):
                    row_content = row_content[1:]
                if row_content.endswith(separator_char):
                    row_content = row_content[:-1]

                cells = [cell.strip() for cell in row_content.split(separator_char)]
                html_parts.append("<tr>")
                for cell in cells:
                    html_parts.append(f"<td>{cell}</td>")
                html_parts.append("</tr>")

        html_parts.append("</tbody>")
        html_parts.append("</table>")

    return "".join(html_parts)


def handle_processing_error(
    exception: Exception, processing_context: str, service_type: str = "document"
) -> str:
    """
    Enhanced error handling with intelligent categorization and user-friendly messaging.

    Args:
        exception: The caught exception
        processing_context: Context string (e.g., "PDF analysis", "Image processing")
        service_type: Type of service for context (e.g., "document", "image")

    Returns:
        str: User-friendly error message
    """
    # Enhanced internal logging with context
    print(f"❌ {processing_context} failed: {exception}")
    print(f"🔍 Error type: {type(exception).__name__}")
    print(f"🔍 Processing context: {processing_context}")
    if hasattr(exception, "__cause__") and exception.__cause__:
        print(f"🔍 Root cause: {exception.__cause__}")

    # Categorize error for better user messaging
    error_str = str(exception).lower()

    # Rate limiting / throttling
    if "rate limit" in error_str or "throttl" in error_str:
        print("📊 Detected: API rate limiting")
        return "Service temporarily busy due to high demand. Please try again in a few minutes."

    # Quota / usage limits
    elif "quota" in error_str or "usage limit" in error_str:
        print("📊 Detected: API quota exceeded")
        return "Service usage limit reached. Please try again later or contact the service owner."

    # Timeout issues
    elif "timeout" in error_str or "timed out" in error_str:
        print("📊 Detected: API timeout")
        if service_type == "document":
            return "Document processing timed out. For large documents, please try processing fewer pages at once."
        else:
            return "Processing timed out. Please try again or contact the service owner."

    # Authentication / credential errors
    elif any(
        term in error_str
        for term in ["credentials", "unauthorized", "access denied", "authentication", "403", "401"]
    ):
        print("📊 Detected: Authentication/credential issue")
        return "Service temporarily unavailable due to authentication error. Please contact the service owner."

    # File corruption / format issues
    elif any(
        term in error_str
        for term in [
            "corrupted",
            "invalid pdf",
            "invalid image",
            "cannot read",
            "unsupported format",
        ]
    ):
        print("📊 Detected: File corruption/format issue")
        if service_type == "document":
            return "Document appears to be corrupted or invalid. Please ensure the file is a valid PDF."
        else:
            return (
                "File format not supported or file is corrupted. Please ensure the file is valid."
            )

    # Memory / size issues
    elif any(term in error_str for term in ["memory", "out of memory", "too large", "file size"]):
        print("📊 Detected: Memory/resource issue")
        if service_type == "document":
            return "Document too large to process. Please try processing fewer pages at once."
        else:
            return "File too large to process. Please try with a smaller file."

    # Generic processing errors
    else:
        print("📊 Detected: General processing error")
        return f"{service_type.title()} processing failed. Please try again or contact the service owner."


def download_file(request: ParseRequest) -> FileData:
    if request.file_bytes is not None:
        file_data = base64.b64decode(request.file_bytes)
        return FileData(file_bytes=file_data, content_type=request.mime_type)
    if request.file_url is not None:
        print(f"getting file input from {request.file_url}")
        file_data, content_type = _download_file(file_url=request.file_url)
        if len(file_data) == 0:
            raise RequestException(message="file is empty")
        return FileData(file_bytes=file_data, content_type=content_type)

    raise RequestException(message="ParseRequest requires either file_bytes or file_url")


def _download_file(file_url: str) -> Tuple[bytes, str]:
    if file_url.startswith("https://"):
        try:
            response = requests.get(file_url, timeout=(5, 60))
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            print(f"HTTP error getting file from URL: {file_url}. {str(e)}")
            raise RequestException(
                message=f"HTTP error accessing file. HTTP status code: {response.status_code}, message: {response.text}"
            ) from e
        except requests.exceptions.Timeout as e:
            print(f"Timeout error getting file from URL: {file_url}. {str(e)}")
            raise RequestException(
                message="Timeout error accessing file. We use a connect timeout of 5 seconds and a read timeout of 60 seconds."
            ) from e
        except requests.exceptions.RequestException as e:
            print(f"Error getting file from URL: {file_url}. {str(e)}")
            raise RequestException(
                message=f"Error accessing file: {str(e)}. Please check the file URL."
            ) from e

        return response.content, response.headers.get("content-type", "")

    raise RequestException(message="file_url must be an HTTPS presigned URL")


# ========================
# NODE-BY-NODE ROUTING CONDITIONS
# ========================


# FILE_CONVERTOR NODE ROUTING
def file_convertor_should_go_to_output_formatter(request) -> bool:
    """PDF/image inputs always continue to OCR or VLM — never terminate here."""
    return False


def file_convertor_should_go_to_vlm_extraction(request) -> bool:
    """File conversion always routes PDF/image inputs to OCR."""
    return False


def file_convertor_should_go_to_ocr(request) -> bool:
    """PDF and image inputs need OCR."""
    return True


# OCR NODE ROUTING (used by every OCR backend after layout extraction)
def ocr_should_go_to_output_formatter(request) -> bool:
    """Go to OutputFormatter if no further processing needed"""
    return not request.key_value_extraction


def _check_has_table_and_figure_and_chart_and_form(
    parse_result: ParseResult,
) -> tuple[bool, bool, bool, bool]:
    """Helper function to check if document has tables and/or figures and/or charts"""
    has_table = False
    has_figure = False
    has_chart = False
    has_form = False
    for page_layout in parse_result.document_layout.pages:
        for element in page_layout.elements:
            if element.fragment_type == PageFragmentType.TABLE:
                has_table = True
            if element.fragment_type == PageFragmentType.TEXT and is_markdown_table(
                element.ocr_text
            ):
                has_table = True
            if element.fragment_type == PageFragmentType.FIGURE:
                has_figure = True
            if element.fragment_type == PageFragmentType.CHART:
                has_chart = True
            if (
                element.fragment_type == PageFragmentType.FORM
                or element.fragment_type == PageFragmentType.KEY_VALUE_REGION
            ):
                has_form = True
            if has_table and has_figure and has_chart and has_form:  # Early exit once we found both
                break
        if has_table and has_figure and has_chart and has_form:
            break
    return has_table, has_figure, has_chart, has_form


def should_route_to_table_merging(request, parse_result: ParseResult) -> bool:
    """Check if we should route to TableMerging task."""
    if not request.table_merging:
        return False

    has_table, _, _, _ = _check_has_table_and_figure_and_chart_and_form(parse_result)
    if not has_table:
        print(
            "Skipping TableMerging: No tables found in document despite table_merging=True request."
        )
        return False

    return True


def route_after_ocr(parse_result: ParseResult, *, log_prefix: str, dots_ocr: bool = False):
    """Dispatch an OCR backend's output to the next pipeline step.

    Every OCR backend ends with the same downstream choice: TableMerging if
    tables exist and ``table_merging`` was requested; otherwise VLM or
    OutputFormatter depending on which post-OCR tasks the request asked for.

    Args:
        parse_result: ParseResult produced by the OCR backend.
        log_prefix: Tag for ``🔀`` log lines (e.g. ``"FULL_PAGE_AZURE"``).
        dots_ocr: If True, use the dots-ocr predicates that additionally
            gate VLM on the actual presence of tables/figures/forms in the
            extracted layout.

    Returns:
        Either a ``ParseResult`` (terminal OutputFormatter step) or a
        Tensorlake future from ``<Task>.run.future(parse_result)``.
    """
    # Lazy imports — these modules import predicates from this file, so a
    # top-level import here would create a cycle.
    from tensorlake_docai.pipeline.output_formatter import format_final_output
    from tensorlake_docai.vlm.cloud import VLMExtractionTask
    from tensorlake_docai.tables.table_merging import TableMerging

    request = parse_result.request

    if should_route_to_table_merging(request, parse_result):
        print(f"🔀 {log_prefix} → TableMerging")
        return TableMerging().run.future(parse_result)

    if dots_ocr:
        go_output = dots_ocr_should_go_to_output_formatter(request, parse_result)
        go_vlm = dots_ocr_should_go_to_vlm_extraction(request, parse_result)
    else:
        go_output = ocr_should_go_to_output_formatter(request)
        go_vlm = ocr_should_go_to_vlm_extraction(request, parse_result)

    if go_output:
        print(f"🔀 {log_prefix} → OutputFormatter")
        return format_final_output(parse_result)

    if go_vlm:
        print(f"🔀 {log_prefix} → VLMExtractionTask")
        return VLMExtractionTask().run.future(parse_result)

    print(f"🔀 {log_prefix} → OutputFormatter")
    return format_final_output(parse_result)


def dots_ocr_should_go_to_output_formatter(request, parse_result: ParseResult) -> bool:
    """Go to OutputFormatter if no further processing needed for dots-ocr."""
    has_table, has_figure, _, has_form = _check_has_table_and_figure_and_chart_and_form(
        parse_result
    )
    return not (request.key_value_extraction and (has_form or has_figure or has_table))


def ocr_should_go_to_vlm_extraction(request, parse_result: ParseResult) -> bool:
    """After OCR, go to VLM if VLM tasks are needed"""
    has_table, has_figure, _, has_form = _check_has_table_and_figure_and_chart_and_form(
        parse_result
    )
    return request.key_value_extraction and (has_figure or has_table or has_form)


def dots_ocr_should_go_to_vlm_extraction(request, parse_result: ParseResult) -> bool:
    """After dots-ocr, go to VLM if key-value extraction has candidate regions."""
    has_table, has_figure, _, has_form = _check_has_table_and_figure_and_chart_and_form(
        parse_result
    )
    return request.key_value_extraction and (has_form or has_figure or has_table)


# VLM_EXTRACTION NODE ROUTING
def vlm_extraction_should_go_to_output_formatter(request) -> bool:
    """VLM enrichment always returns to OutputFormatter."""
    return True


async def stream_with_timeout(stream, timeout_seconds: int = 120):
    """
    Yields chunks from stream with a per-chunk timeout.
    """
    # Convert the stream to an asynchronous iterator
    iterator = aiter(stream)

    try:
        while True:
            try:
                chunk = await asyncio.wait_for(anext(iterator), timeout=timeout_seconds)
                yield chunk
            except StopAsyncIteration:
                break
            except asyncio.TimeoutError:
                raise asyncio.TimeoutError(
                    f"No response chunk received within {timeout_seconds} seconds"
                ) from None
    finally:
        if hasattr(iterator, "aclose"):
            await iterator.aclose()


def update_progress_if_needed(
    chunk_count: int, accumulated_chars: int, start_time: float, last_update_time: float
) -> float:
    """
    Update progress to keep function-level timeout alive.
    Returns the new last_update_time if update was performed, otherwise returns the input last_update_time.
    """
    from tensorlake.applications import RequestContext

    try:
        ctx = RequestContext.get()
    except Exception:
        return last_update_time

    # Update every 5 chunks or every 30 seconds
    current_time = time.time()
    if chunk_count % 5 == 0 or (current_time - last_update_time) >= 30:
        elapsed = current_time - start_time
        ctx.progress.update(
            current=accumulated_chars,
            total=accumulated_chars + 1000,  # Estimated total
            message=f"Streaming response ({accumulated_chars} chars, {elapsed:.0f}s)",
        )
        return current_time
    return last_update_time
