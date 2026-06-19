# SPDX-License-Identifier: Apache-2.0
from typing import Any, Dict, Optional

from tensorlake_docai.pipeline.api import ParseRequest, Usage
from tensorlake_docai.models.layout_objects import DocumentLayout
from pydantic import BaseModel


class FormFillingResult(BaseModel):
    filled_pdf_base64: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ParseResult(BaseModel):
    document_layout: Optional[DocumentLayout] = None
    form_filling_result: Optional[FormFillingResult] = None
    structured_outputs_by_page: Optional[Dict[int, Any]] = None
    request: ParseRequest
    usage: Optional[Usage] = None


class FileData(BaseModel):
    file_bytes: Optional[bytes] = None
    content_type: Optional[str] = None
