# SPDX-License-Identifier: Apache-2.0
from enum import Enum
from typing import Dict, List, Literal, Optional, Set, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field

# PDF and image inputs accepted by this service (see file_converter.py).
SUPPORTED_MIME_TYPES = frozenset(
    {
        "application/pdf",
        "image/png",
        "image/jpg",
        "image/jpeg",
        "image/heif",
        "image/heic",
    }
)


### OUTPUT API FROM THE WORKFLOW #####
class PageFragmentType(str, Enum):
    """
    Type of a page fragment.
    """

    SECTION_HEADER = "section_header"
    TITLE = "title"

    TEXT = "text"
    TABLE = "table"
    FIGURE = "figure"
    CHART = "chart"
    FORMULA = "formula"
    FORM = "form"
    KEY_VALUE_REGION = "key_value_region"
    DOCUMENT_INDEX = "document_index"
    LIST_ITEM = "list_item"

    TABLE_CAPTION = "table_caption"
    FIGURE_CAPTION = "figure_caption"
    FORMULA_CAPTION = "formula_caption"

    PAGE_FOOTER = "page_footer"
    PAGE_HEADER = "page_header"
    PAGE_NUMBER = "page_number"
    SIGNATURE = "signature"

    TRACKED_CHANGES = "tracked_changes"
    COMMENTS = "comments"
    BARCODE = "barcode"


class Text(BaseModel):
    content: str
    html: Optional[str] = None


class SectionHeader(BaseModel):
    content: str
    level: int  # 1 for #, 2 for ##, 3 for ###, etc.


class Signature(BaseModel):
    content: Optional[str] = None


class ListItem(BaseModel):
    content: str


class Table(BaseModel):
    content: str
    summary: Optional[str] = None
    html: Optional[str] = None
    markdown: Optional[str] = None
    table_checked: Optional[bool] = False  # Checked and processed through table correction


class Figure(BaseModel):
    content: str
    summary: Optional[str] = None
    image_base64: Optional[str] = None  # Base64-encoded image data for rendering


class Chart(BaseModel):
    content: str
    image_base64: Optional[str] = None  # Base64-encoded image data for rendering


class PageFragment(BaseModel):
    fragment_type: PageFragmentType
    content: Union[Text, Table, Figure, Chart, ListItem, Signature, SectionHeader]
    reading_order: Optional[int] = None
    bbox: Optional[dict[str, float]] = None
    ref_id: Optional[str] = None  # Format: page.reading_order or page.reading_order.cell_index


class Page(BaseModel):
    """
    Page in a document.
    """

    page_number: int
    page_fragments: Optional[List[PageFragment]] = []
    dimensions: Optional[Tuple[int, int]] = None
    page_dimensions: Optional[Dict[str, int]] = None
    page_class: Optional[Union[str, List[str]]] = None
    classification_reason: Optional[str] = None
    classification_confidence: Optional[float] = None


class Chunk(BaseModel):
    content: str
    page_number: int  # For backward compatibility - the starting page
    page_numbers: Optional[List[int]] = None  # All pages this chunk spans
    element_ids: Optional[List[str]] = (
        None  # ref_ids of elements in this chunk (e.g., ["2.5", "2.6", "3.1"])
    )


class PageClass(BaseModel):
    page_numbers: List[int]
    classification_reasons: Optional[Dict[int, str]] = None
    classification_confidences: Optional[Dict[int, float]] = None
    page_class: str


class MergeTableActions(BaseModel):
    pages: List[int]
    ref_ids: Optional[List[str]] = None
    target_columns: Optional[int] = None


class MergedTable(BaseModel):
    merged_table_id: str
    merged_table_html: str
    start_page: int
    end_page: int
    pages_merged: int
    summary: Optional[str] = None
    merge_actions: MergeTableActions


class ParsedDocument(BaseModel):
    parsed_pages_count: Optional[int] = None
    pages: Optional[List[Page]] = None
    chunks: List[Chunk]
    merged_tables: Optional[List[MergedTable]] = None
    total_pages: Optional[int] = None
    page_classes: Optional[List[PageClass]] = None
    document_markdown: Optional[str] = None  # Full document markdown representation


class Usage(BaseModel):
    pages_parsed: int
    ocr_input_tokens_used: Optional[int] = None
    ocr_output_tokens_used: Optional[int] = None
    extraction_input_tokens_used: Optional[int] = None
    extraction_output_tokens_used: Optional[int] = None
    summarization_input_tokens_used: Optional[int] = None
    summarization_output_tokens_used: Optional[int] = None
    header_correction_input_tokens_used: Optional[int] = None
    header_correction_output_tokens_used: Optional[int] = None


class ParsedDocumentRef(BaseModel):
    document: Optional[Dict] = None
    usage: Optional[Usage] = None


class QuotaResourceType(str, Enum):
    PAGES_PARSED = "pages_parsed"


class ResourceQuotaRequest(BaseModel):
    # this is to allow the use of alias for the fields
    model_config = ConfigDict(populate_by_name=True)

    resource_type: QuotaResourceType = Field(alias="resourceType")
    remaining_quota: int = Field(
        alias="remainingQuota",
        description="Remaining quota for this resource type. Use -1 for unlimited quota.",
    )


class OrganizationQuotaRequest(BaseModel):
    # this is to allow the use of alias for the fields
    model_config = ConfigDict(populate_by_name=True)

    organization_id: str = Field(alias="organizationId")
    quotas: List[ResourceQuotaRequest]


##### REQUEST API INTO THE WORKFLOW #####
class PageClassDefinition(BaseModel):
    class_name: str
    description: str


class ClassificationRequest(BaseModel):
    """
    Request for page classification.

    classification_type:
        - "multi-label": Each page can belong to multiple classes simultaneously (default).
        - "multi-class": Each page can belong to only one class.
    """

    class_definitions: List[PageClassDefinition]
    classification_type: Literal["multi_label", "multi_class"] = Field(
        default="multi_label",
        description=(
            "Type of classification to perform. "
            "'multi-label' allows each page to have multiple classes. "
            "'multi-class' restricts each page to a single class."
        ),
    )


class ParseRequest(BaseModel):
    file_bytes: Optional[str] = None
    file_url: Optional[str] = None

    pages_to_parse: Optional[List[int]] = None
    file_name: str
    mime_type: str
    skew_correction: bool = False
    detect_barcode: bool = False
    debug: bool = False
    chunk_strategy: Optional[str] = None
    table_parsing_strategy: Optional[Literal["tsr", "vlm"]] = "vlm"
    table_output_mode: Optional[Literal["html", "json", "markdown"]] = "markdown"
    ocr_model: Optional[Literal["dots-ocr"]] = "dots-ocr"
    page_classification_request: Optional[ClassificationRequest] = None
    disable_layout_detection: Optional[bool] = False
    table_summarization: Optional[bool] = False
    table_summarization_prompt: Optional[str] = None
    table_merging: bool = False
    figure_summarization: Optional[bool] = False
    figure_summarization_prompt: Optional[str] = None
    figure_ocr_prompt: Optional[str] = None  # For automatic figure OCR in the `dots-ocr` path
    # This is to make the full page image in table and figure summarization optional
    chart_extraction: Optional[bool] = False
    key_value_extraction: Optional[bool] = False
    include_full_page_image: Optional[bool] = False
    ignore_sections: Optional[Set[PageFragmentType]] = None
    org_quota: Optional[OrganizationQuotaRequest] = None
    xpage_header_detection: bool = False
    include_images: Optional[bool] = False
