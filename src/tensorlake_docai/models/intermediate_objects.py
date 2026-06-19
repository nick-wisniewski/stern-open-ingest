# SPDX-License-Identifier: Apache-2.0
from typing import Optional

from tensorlake_docai.pipeline.api import ParseRequest, Usage
from tensorlake_docai.models.layout_objects import DocumentLayout
from pydantic import BaseModel


class ParseResult(BaseModel):
    document_layout: Optional[DocumentLayout] = None
    request: ParseRequest
    usage: Optional[Usage] = None


class FileData(BaseModel):
    file_bytes: Optional[bytes] = None
    content_type: Optional[str] = None
