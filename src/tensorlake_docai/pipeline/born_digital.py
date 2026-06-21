# SPDX-License-Identifier: Apache-2.0
"""CPU text-layer routing and extraction for born-digital PDF pages."""

from __future__ import annotations

import os
from typing import Literal

from pydantic import BaseModel
from tensorlake.applications import Retries, function

from tensorlake_docai.models.intermediate_objects import ParseResult
from tensorlake_docai.models.layout_objects import PageLayout, PageLayoutElement
from tensorlake_docai.ocr import resolve_ocr_backend
from tensorlake_docai.pipeline.api import PageFragmentType
from tensorlake_docai.pipeline.routing import route_after_ocr
from tensorlake_docai.vlm.workflow_images import file_convertion_image

PageRoute = Literal["born_digital", "needs_ocr", "needs_ocr_for_tables"]


class PageClassification(BaseModel):
    page_number: int
    route: PageRoute
    reason: str
    text_chars: int = 0
    image_area_ratio: float = 0.0


def selected_pages(total_pages: int, pages_to_parse: list[int] | None = None) -> list[int]:
    pages = pages_to_parse or list(range(1, total_pages + 1))
    return [page for page in sorted(set(pages)) if 1 <= page <= total_pages]


def classify_pdf_pages(
    file_bytes: bytes,
    *,
    total_pages: int,
    pages_to_parse: list[int] | None = None,
) -> list[PageClassification]:
    """Conservatively decide which PDF pages can skip GPU OCR."""

    import fitz

    wanted = selected_pages(total_pages, pages_to_parse)
    decisions: list[PageClassification] = []

    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page_number in wanted:
            page = doc.load_page(page_number - 1)
            text = page.get_text("text").strip()
            text_chars = len("".join(text.split()))
            page_area = max(float(page.rect.width * page.rect.height), 1.0)
            image_area = 0.0

            for image in page.get_images(full=True):
                bbox = page.get_image_bbox(image)
                if not bbox.is_empty:
                    image_area += float(bbox.width * bbox.height)

            image_area_ratio = min(image_area / page_area, 1.0)
            block_count = len(page.get_text("blocks") or [])

            if text_chars < 40:
                route: PageRoute = "needs_ocr"
                reason = "too little extractable text"
            elif image_area_ratio > 0.65:
                route = "needs_ocr"
                reason = "image-dominant page"
            elif _looks_like_hard_table(text, block_count):
                route = "needs_ocr_for_tables"
                reason = "table-like text layer"
            else:
                route = "born_digital"
                reason = "usable PDF text layer"

            decisions.append(
                PageClassification(
                    page_number=page_number,
                    route=route,
                    reason=reason,
                    text_chars=text_chars,
                    image_area_ratio=image_area_ratio,
                )
            )

    return decisions


def _looks_like_hard_table(text: str, block_count: int) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 8:
        return False

    numeric_lines = sum(1 for line in lines if any(char.isdigit() for char in line))
    dense_blocks = block_count >= 40
    tabular_separators = sum(1 for line in lines if line.count("  ") >= 3 or "\t" in line)
    return dense_blocks and numeric_lines >= len(lines) * 0.5 and tabular_separators >= 3


def ocr_pages_from_classification(decisions: list[PageClassification]) -> list[int]:
    return [
        decision.page_number
        for decision in decisions
        if decision.route in {"needs_ocr", "needs_ocr_for_tables"}
    ]


def cpu_text_pages_for_request(total_pages: int, request) -> list[int]:
    if request.mime_type != "application/pdf":
        return []

    wanted = selected_pages(total_pages, request.pages_to_parse)
    ocr_pages = set(request.ocr_pages or [])
    return [page for page in wanted if page not in ocr_pages]


def extract_pdf_text_layer_pages(file_bytes: bytes, page_numbers: list[int]) -> list[PageLayout]:
    import fitz

    pages: list[PageLayout] = []
    wanted = set(page_numbers)

    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page_number in sorted(wanted):
            page = doc.load_page(page_number - 1)
            elements: list[PageLayoutElement] = []

            for reading_order, block in enumerate(page.get_text("blocks") or []):
                x0, y0, x1, y1, text, *_rest = block
                content = str(text).strip()
                if not content:
                    continue
                if x1 <= x0 or y1 <= y0:
                    continue

                elements.append(
                    PageLayoutElement(
                        bbox=(float(x0), float(y0), float(x1), float(y1)),
                        fragment_type=PageFragmentType.TEXT,
                        score=1.0,
                        reading_order=reading_order,
                        ref_id=f"{page_number}.{reading_order}",
                        ocr_text=content,
                        markdown=content,
                    )
                )

            pages.append(
                PageLayout(
                    elements=elements,
                    shape=(round(page.rect.width), round(page.rect.height)),
                    page_number=page_number,
                    page_dimensions={
                        "width": round(page.rect.width),
                        "height": round(page.rect.height),
                    },
                )
            )

    return pages


@function(
    description="Extract born-digital PDF text layers before optional GPU OCR.",
    image=file_convertion_image,
    timeout=30 * 60,
    cpu=2,
    memory=5,
    ephemeral_disk=10,
    retries=Retries(max_retries=2),
    max_containers=200,
    min_containers=int(os.getenv("TENSORLAKE_MIN_CONTAINERS", "0")),
)
def extract_born_digital_pages(parse_result: ParseResult) -> ParseResult | dict:
    request = parse_result.request
    total_pages = parse_result.document_layout.total_pages
    page_numbers = cpu_text_pages_for_request(total_pages, request)

    if page_numbers:
        print(f"Extracting born-digital text layer for pages: {page_numbers}")
        parse_result.document_layout.pages.extend(
            extract_pdf_text_layer_pages(request.file_bytes, page_numbers)
        )
        parse_result.document_layout.pages.sort(key=lambda page: page.page_number)

    if request.ocr_pages:
        backend_cls = resolve_ocr_backend(request.ocr_model)
        print(
            f"Born-digital extraction complete; routing OCR pages {request.ocr_pages} "
            f"to {backend_cls.__name__}"
        )
        return backend_cls().run.future(parse_result)

    return route_after_ocr(parse_result, log_prefix="BornDigitalExtraction")
