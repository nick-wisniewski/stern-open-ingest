# SPDX-License-Identifier: Apache-2.0
"""Accepted input MIME types for the parsing service."""

import pytest

from tensorlake.applications import RequestError as RequestException
from tensorlake_docai.pipeline.api import SUPPORTED_MIME_TYPES
from tensorlake_docai.pipeline.file_converter import validate_supported_mime_type


def test_supported_mime_types_match_product_list():
    assert SUPPORTED_MIME_TYPES == frozenset(
        {
            "application/pdf",
            "image/png",
            "image/jpg",
            "image/jpeg",
            "image/heif",
            "image/heic",
        }
    )


def test_validate_accepts_listed_mime_types():
    for mime in SUPPORTED_MIME_TYPES:
        assert validate_supported_mime_type(mime) == mime


def test_validate_accepts_extension_fallback():
    assert validate_supported_mime_type("application/octet-stream", "scan.heic") == "image/heic"


def test_validate_rejects_unsupported_mime():
    with pytest.raises(RequestException, match="Unsupported file type"):
        validate_supported_mime_type("text/plain", "notes.txt")
