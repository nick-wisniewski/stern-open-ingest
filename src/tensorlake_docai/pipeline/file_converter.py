# SPDX-License-Identifier: Apache-2.0
import io
import os
import traceback
from pathlib import Path
from typing import Optional

from tensorlake.applications import application, function, Retries
from tensorlake.applications import RequestError as RequestException
from tensorlake_docai.vlm.cloud import VLMExtractionTask
from tensorlake_docai.pipeline.output_formatter import format_final_output
from tensorlake_docai.vlm.workflow_images import file_convertion_image
from tensorlake_docai.pipeline.api import (
    ParseRequest,
    Usage,
    QuotaResourceType,
    SUPPORTED_MIME_TYPES,
)
from tensorlake_docai.ocr import resolve_ocr_backend
from tensorlake_docai.pipeline.routing import (
    FILE_TYPE_MAPPING,
    download_file,
    file_convertor_should_go_to_output_formatter,
    file_convertor_should_go_to_vlm_extraction,
    file_convertor_should_go_to_ocr,
)
from tensorlake_docai.models.intermediate_objects import ParseResult
from tensorlake_docai.models.layout_objects import DocumentLayout

SECRETS = [
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_REGION",
]

_EXTENSION_TO_MIME = {
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".jpg": "image/jpg",
    ".jpeg": "image/jpeg",
    ".heif": "image/heif",
    ".heic": "image/heic",
}


def _mime_from_filename(file_name: str) -> Optional[str]:
    if not file_name:
        return None
    return _EXTENSION_TO_MIME.get(Path(file_name).suffix.lower())


def is_supported_mime_type(mime_type: str) -> bool:
    return mime_type in SUPPORTED_MIME_TYPES


def validate_supported_mime_type(mime_type: str, file_name: str = "") -> str:
    """Return a supported MIME type or raise RequestException."""
    if is_supported_mime_type(mime_type):
        return mime_type

    # Accept common extension hints when the caller supplied a non-standard MIME string.
    fallback = _mime_from_filename(file_name)
    if fallback and is_supported_mime_type(fallback):
        return fallback

    supported = ", ".join(sorted(SUPPORTED_MIME_TYPES))
    raise RequestException(
        message=(
            f"Unsupported file type {mime_type!r}. "
            f"This service accepts PDF and image inputs only: {supported}."
        )
    )


def detect_mime_type_from_content(
    file_bytes: bytes, file_name: str = None, url_content_type: str = None
) -> str:
    """Detect MIME type from file content and ensure it is supported."""
    try:
        import magic

        detected_mime = magic.from_buffer(file_bytes, mime=True)

        if not is_supported_mime_type(detected_mime):
            file_extension = FILE_TYPE_MAPPING.get(detected_mime, None)

            if file_extension is None and file_name:
                import mimetypes

                detected_mime, _ = mimetypes.guess_type(file_name)
                file_extension = FILE_TYPE_MAPPING.get(detected_mime, None)

            if file_extension is None and url_content_type:
                detected_mime = url_content_type

        if detected_mime is None:
            raise RequestException(
                message=(
                    "Unable to detect a supported file type. "
                    "Please provide a PDF or image (PNG, JPEG, HEIF, HEIC)."
                )
            )

        return validate_supported_mime_type(detected_mime, file_name or "")
    except RequestException:
        raise
    except Exception as e:
        print(traceback.format_exc())
        print(f"DEBUG: MIME detection failed: {e}")
        raise RequestException(
            message=(
                "Unable to detect file type. Please upload a valid PDF or image file. Error: "
                + str(e)
            )
        )


def count_document_pages(file_bytes: bytes, mime_type: str, file_name: str = "") -> int:
    """Count pages for PDFs and single-page images."""
    print(f"DEBUG: count_document_pages called with mime_type: {mime_type}, file_name: {file_name}")

    mime_type = validate_supported_mime_type(mime_type, file_name)

    if mime_type == "application/pdf":
        try:
            from pypdf import PdfReader

            return len(PdfReader(io.BytesIO(file_bytes)).pages)
        except Exception as e:
            print(f"DEBUG: Failed to read PDF for page counting: {e}")
            raise RequestException(
                message=(
                    "Unable to open the PDF document. Please ensure the file is a valid, "
                    "non-corrupted PDF. Error: " + str(e)
                )
            )

    if mime_type.startswith("image/"):
        print("DEBUG: Treating as single-page image file")
        return 1

    raise RequestException(message=f"Unsupported file type for page counting: {mime_type}")


def validate_quota(request: ParseRequest, total_document_pages: int) -> None:
    """
    Validate that the request doesn't exceed any quota limits.
    Raises an exception if any quota is exceeded.

    Note: A remaining_quota value of -1 indicates unlimited quota (no validation).
    """
    if request.org_quota is None:
        return

    if request.pages_to_parse is not None:
        pages_to_process = len(request.pages_to_parse)
    else:
        pages_to_process = total_document_pages

    for quota in request.org_quota.quotas:
        if quota.remaining_quota == -1:
            print(
                f"Skipping quota validation for {quota.resource_type.value} because it is unlimited"
            )
            continue

        if quota.resource_type == QuotaResourceType.PAGES_PARSED:
            if pages_to_process > quota.remaining_quota:
                raise RequestException(
                    message=f"Quota exceeded: Requested {pages_to_process} pages but only "
                    f"{quota.remaining_quota} pages remaining for {quota.resource_type.value}"
                )


def process_file_from_s3_or_url(request: ParseRequest) -> None:
    """Download (if needed), detect MIME type, and validate the input file."""
    print("DEBUG: Processing file input")
    file_data = download_file(request)
    detected_mime = detect_mime_type_from_content(
        file_data.file_bytes, request.file_name, file_data.content_type
    )
    request.mime_type = detected_mime
    request.file_bytes = file_data.file_bytes


@application()
@function(
    description="Validate and normalize PDF/image inputs for the parsing pipeline.",
    image=file_convertion_image,
    secrets=SECRETS,
    timeout=30 * 60,
    cpu=2,
    memory=5,
    ephemeral_disk=10,
    retries=Retries(max_retries=2),
    max_containers=200,
    min_containers=int(os.getenv("TENSORLAKE_MIN_CONTAINERS", "0")),
)
def normalize_file_type_and_upload(raw_request: dict) -> ParseResult | dict:
    print(f"DEBUG: raw_request keys: {list(raw_request.keys())}")
    print(f"DEBUG: raw_request mime_type: {repr(raw_request.get('mime_type'))}")
    print(f"DEBUG: raw_request file_name: {repr(raw_request.get('file_name'))}")
    print(f"DEBUG: raw_request file_url: {repr(raw_request.get('file_url'))}")
    print(
        f"DEBUG: raw_request file_bytes: {repr(raw_request.get('file_bytes', 'NOT_PROVIDED')[:50] if raw_request.get('file_bytes') else None)}"
    )

    request = ParseRequest.model_validate(raw_request)
    print(f"DEBUG: request keys: {list(request.model_dump().keys())}")

    process_file_from_s3_or_url(request)
    print(f"DEBUG: request file_bytes length: {len(request.file_bytes)}")

    try:
        total_document_pages = count_document_pages(
            request.file_bytes, request.mime_type, request.file_name
        )
        print(f"Document has {total_document_pages} total pages")
    except RequestException:
        raise
    except Exception as e:
        print(f"Warning: Could not count document pages: {e}")
        total_document_pages = 1

    validate_quota(request, total_document_pages)

    if request.pages_to_parse:
        valid_pages = [p for p in request.pages_to_parse if 1 <= p <= total_document_pages]
        invalid_pages = [p for p in request.pages_to_parse if p < 1 or p > total_document_pages]

        if invalid_pages:
            print(
                f"Warning: Ignoring invalid pages {invalid_pages} "
                f"(document has {total_document_pages} pages)"
            )

        if not valid_pages:
            first_invalid = sorted(invalid_pages)[0]
            raise RequestException(
                message=(
                    f"Invalid page range specified. Document has {total_document_pages} pages, "
                    f"but page {first_invalid} was requested. "
                    f"Please specify pages 1-{total_document_pages}."
                )
            )

        request.pages_to_parse = valid_pages
        print(f"Processing valid pages: {valid_pages}")

    usage = Usage(
        pages_parsed=0,
        extraction_input_tokens_used=0,
        extraction_output_tokens_used=0,
        summarization_input_tokens_used=0,
        summarization_output_tokens_used=0,
        header_correction_input_tokens_used=0,
        header_correction_output_tokens_used=0,
    )
    print(f"Accepted input mime_type={request.mime_type!r}")
    parse_result = ParseResult(
        document_layout=DocumentLayout(
            pages=[], scale_factor=1.0, total_pages=total_document_pages
        ),
        request=request,
        usage=usage,
    )

    if file_convertor_should_go_to_output_formatter(request):
        print("🔀 FILE_CONVERTOR → OutputFormatter")
        return format_final_output(parse_result)

    if file_convertor_should_go_to_vlm_extraction(request):
        print("🔀 FILE_CONVERTOR → VLMExtractionTask")
        return VLMExtractionTask().run.future(parse_result)

    if file_convertor_should_go_to_ocr(request):
        backend_cls = resolve_ocr_backend(request.ocr_model)
        print(f"🔀 FILE_CONVERTOR → {backend_cls.__name__} (ocr_model={request.ocr_model!r})")
        return backend_cls().run.future(parse_result)

    print("🔀 FILE_CONVERTOR → OutputFormatter (fallback)")
    return format_final_output(parse_result)
