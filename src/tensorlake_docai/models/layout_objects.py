# SPDX-License-Identifier: Apache-2.0
from pydantic import BaseModel
from typing import List, Tuple, Optional, Union
from tensorlake_docai.pipeline.api import PageFragmentType, MergedTable


class PageLayoutElement(BaseModel):
    bbox: Tuple[float, float, float, float]
    fragment_type: PageFragmentType
    score: float
    reading_order: int = -1
    ref_id: Optional[str] = None  # Format: page.reading_order (e.g., "1.16")
    ocr_text: Optional[str] = None
    llm_summary: Optional[str] = None
    html: Optional[str] = None
    markdown: Optional[str] = None
    hierarchy_level: Optional[int] = None  # For section headers: 0=top level, 1=subsection, etc.
    image_base64: Optional[str] = None  # Base64-encoded image data for FIGURE elements
    table_checked: Optional[bool] = False  # Indicates if the element has been checked/corrected


class PageLayout(BaseModel):
    elements: List[PageLayoutElement]
    shape: Tuple[int, int]
    page_number: int
    page_class: Optional[Union[str, List[str]]] = None
    classification_reason: Optional[str] = None
    classification_confidence: Optional[float] = None
    page_dimensions: Optional[dict] = None  # record {"width": int, "height": int} from shape


class DocumentLayout(BaseModel):
    pages: List[PageLayout]
    scale_factor: float
    total_pages: int
    merged_tables: Optional[List[MergedTable]] = None
