# SPDX-License-Identifier: Apache-2.0
import os
import subprocess
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Iterable

from tensorlake.applications import RequestContext, Retries, cls, function
from tensorlake.applications import RequestError as RequestException
from tensorlake_docai.models.intermediate_objects import ParseResult
from tensorlake_docai.models.layout_objects import PageLayout, PageLayoutElement
from tensorlake_docai.ocr.utils import BatchProcessor
from tensorlake_docai.pipeline.api import PageFragmentType
from tensorlake_docai.pipeline.routing import route_after_ocr
from tensorlake_docai.pipeline.simple_page_creator import ImageDimensions
from tensorlake_docai.vlm.workflow_images import paddle_ocr_vl_image

PADDLE_OCR_VL_MEMORY_IN_GB = int(os.getenv("PADDLE_OCR_VL_MEMORY_IN_GB", "24"))
PADDLE_OCR_VL_GPU_MODELS = os.getenv("PADDLE_OCR_VL_GPU_MODELS", "L4,A10G").split(",")
PADDLE_OCR_VL_SERVER_URL = os.getenv("PADDLE_OCR_VL_SERVER_URL", "http://127.0.0.1:8118/v1")
PADDLE_OCR_VL_REC_BACKEND = os.getenv("PADDLE_OCR_VL_REC_BACKEND", "vllm-server")
PADDLE_OCR_VL_DEVICE = os.getenv("PADDLE_OCR_VL_DEVICE")


def _cuda_is_available() -> bool:
    try:
        import torch

        if torch.cuda.is_available():
            return True
    except Exception:
        pass

    try:
        import paddle

        if paddle.device.cuda.device_count() > 0:
            return True
    except Exception:
        pass

    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


def _unwrap_paddle_result(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        raw = result
    elif hasattr(result, "json"):
        raw = result.json
    else:
        raise RequestException(message=f"Unsupported PaddleOCR-VL result type: {type(result)!r}")

    if not isinstance(raw, dict):
        raise RequestException(message="PaddleOCR-VL returned a non-dictionary result.")

    nested = raw.get("res")
    data = nested if isinstance(nested, dict) else raw
    if hasattr(result, "markdown"):
        data["_markdown_result"] = result.markdown
    return data


def _coerce_bbox(value: Any) -> tuple[float, float, float, float] | None:
    if value is None:
        return None

    if hasattr(value, "tolist"):
        value = value.tolist()

    if isinstance(value, dict):
        value = [
            value.get("x0", value.get("left")),
            value.get("y0", value.get("top")),
            value.get("x1", value.get("right")),
            value.get("y1", value.get("bottom")),
        ]

    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None

    try:
        x0, y0, x1, y1 = (float(v) for v in value)
    except (TypeError, ValueError):
        return None

    if x1 <= x0 or y1 <= y0:
        return None

    return x0, y0, x1, y1


def _scale_bbox(
    bbox: tuple[float, float, float, float],
    image_size: tuple[int, int],
    pdf_size: tuple[int, int],
) -> tuple[float, float, float, float]:
    img_w, img_h = image_size
    pdf_w, pdf_h = pdf_size
    sx = (pdf_w / img_w) if img_w else 1.0
    sy = (pdf_h / img_h) if img_h else 1.0
    return bbox[0] * sx, bbox[1] * sy, bbox[2] * sx, bbox[3] * sy


def map_paddle_label_to_fragment_type(label: str | None) -> PageFragmentType:
    normalized = (label or "").strip().lower().replace("-", "_").replace(" ", "_")
    mapping = {
        "doc_title": PageFragmentType.TITLE,
        "title": PageFragmentType.TITLE,
        "paragraph_title": PageFragmentType.SECTION_HEADER,
        "section_header": PageFragmentType.SECTION_HEADER,
        "header": PageFragmentType.PAGE_HEADER,
        "page_header": PageFragmentType.PAGE_HEADER,
        "footer": PageFragmentType.PAGE_FOOTER,
        "page_footer": PageFragmentType.PAGE_FOOTER,
        "page_number": PageFragmentType.PAGE_NUMBER,
        "text": PageFragmentType.TEXT,
        "paragraph": PageFragmentType.TEXT,
        "list": PageFragmentType.LIST_ITEM,
        "list_item": PageFragmentType.LIST_ITEM,
        "table": PageFragmentType.TABLE,
        "table_title": PageFragmentType.TABLE_CAPTION,
        "table_caption": PageFragmentType.TABLE_CAPTION,
        "image": PageFragmentType.FIGURE,
        "figure": PageFragmentType.FIGURE,
        "chart": PageFragmentType.CHART,
        "formula": PageFragmentType.FORMULA,
        "formula_caption": PageFragmentType.FORMULA_CAPTION,
        "seal": PageFragmentType.FIGURE,
    }
    return mapping.get(normalized, PageFragmentType.TEXT)


def _markdown_from_html(html: str) -> str:
    try:
        from markdownify import markdownify as md

        return md(html)
    except Exception:
        return html


def _iter_parsing_blocks(result: dict[str, Any]) -> Iterable[dict[str, Any]]:
    parsing_blocks = result.get("parsing_res_list")
    if isinstance(parsing_blocks, list):
        yield from (block for block in parsing_blocks if isinstance(block, dict))
        return

    layout_boxes = result.get("layout_det_res", {}).get("boxes")
    if isinstance(layout_boxes, list):
        for idx, box in enumerate(layout_boxes):
            if not isinstance(box, dict):
                continue
            yield {
                "block_bbox": box.get("coordinate"),
                "block_label": box.get("label"),
                "block_content": "",
                "block_id": idx,
                "block_order": idx,
                "score": box.get("score", 1.0),
            }


def _markdown_text_from_result(result: dict[str, Any]) -> str:
    markdown = result.get("_markdown_result") or result.get("markdown")
    if not isinstance(markdown, dict):
        return ""

    markdown_texts = markdown.get("markdown_texts")
    if isinstance(markdown_texts, str):
        return markdown_texts
    if isinstance(markdown_texts, list):
        return "\n\n".join(str(text) for text in markdown_texts if text)

    text = markdown.get("text")
    return text if isinstance(text, str) else ""


def paddle_result_to_page_layout(
    result: Any,
    *,
    page_number: int,
    image_size: tuple[int, int],
    pdf_size: tuple[int, int],
) -> PageLayout:
    data = _unwrap_paddle_result(result)
    elements: list[PageLayoutElement] = []

    for idx, block in enumerate(_iter_parsing_blocks(data)):
        bbox = _coerce_bbox(block.get("block_bbox") or block.get("bbox"))
        if bbox is None:
            continue

        label = block.get("block_label") or block.get("label")
        fragment_type = map_paddle_label_to_fragment_type(label)
        content = str(block.get("block_content") or block.get("text") or "")
        reading_order = block.get("block_order")
        if reading_order is None:
            reading_order = block.get("block_id", idx)

        try:
            reading_order = int(reading_order)
        except (TypeError, ValueError):
            reading_order = idx

        pdf_bbox = _scale_bbox(bbox, image_size, pdf_size)
        html = content if fragment_type == PageFragmentType.TABLE and "<table" in content else None
        markdown = _markdown_from_html(content) if html else content
        hierarchy_level = 0 if fragment_type == PageFragmentType.TITLE else None

        elements.append(
            PageLayoutElement(
                bbox=pdf_bbox,
                fragment_type=fragment_type,
                score=float(block.get("score", 1.0) or 1.0),
                reading_order=reading_order,
                ref_id=f"{page_number}.{reading_order}",
                ocr_text=content,
                html=html,
                markdown=markdown,
                hierarchy_level=hierarchy_level,
            )
        )

    if not any((element.ocr_text or "").strip() for element in elements):
        markdown_text = _markdown_text_from_result(data).strip()
        if markdown_text:
            elements = [
                PageLayoutElement(
                    bbox=(0.0, 0.0, float(pdf_size[0]), float(pdf_size[1])),
                    fragment_type=PageFragmentType.TEXT,
                    score=1.0,
                    reading_order=0,
                    ref_id=f"{page_number}.0",
                    ocr_text=markdown_text,
                    markdown=markdown_text,
                )
            ]

    elements.sort(key=lambda element: element.reading_order)
    return PageLayout(
        elements=elements,
        shape=(int(pdf_size[0]), int(pdf_size[1])),
        page_number=page_number,
        page_dimensions={"width": int(pdf_size[0]), "height": int(pdf_size[1])},
    )


@cls()
class PaddleOCRVLTask(BatchProcessor):
    def __init__(self):
        super().__init__(memory_gb=PADDLE_OCR_VL_MEMORY_IN_GB / 8)
        self.pipeline = None

    def _initialize_pipeline(self):
        if self.pipeline is not None:
            return

        from paddleocr import PaddleOCRVL

        print(
            "Initializing PaddleOCR-VL client "
            f"(vl_rec_backend={PADDLE_OCR_VL_REC_BACKEND!r}, "
            f"server={PADDLE_OCR_VL_SERVER_URL!r}, "
            f"device={PADDLE_OCR_VL_DEVICE!r})"
        )
        kwargs = {
            "vl_rec_backend": PADDLE_OCR_VL_REC_BACKEND,
            "vl_rec_server_url": PADDLE_OCR_VL_SERVER_URL,
            "format_block_content": True,
        }
        if PADDLE_OCR_VL_DEVICE:
            kwargs["device"] = PADDLE_OCR_VL_DEVICE

        try:
            self.pipeline = PaddleOCRVL(**kwargs)
        except Exception as e:
            raise RequestException(
                message=(
                    "Failed to initialize PaddleOCR-VL client pipeline: "
                    f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
                )
            ) from e

    async def process_batch(self, processing_batch, batch_number):
        self._initialize_pipeline()

        page_numbers = processing_batch.page_numbers
        page_images = processing_batch.payload
        original_sizes = processing_batch.original_sizes

        with tempfile.TemporaryDirectory(prefix="paddle_ocr_vl_") as tmpdir:
            input_paths = []
            for page_number, image in zip(page_numbers, page_images):
                path = Path(tmpdir) / f"page_{page_number}.png"
                image.save(path)
                input_paths.append(str(path))

            start = time.time()
            try:
                predictions = list(self.pipeline.predict(input_paths))
            except Exception as e:
                raise RequestException(
                    message=(
                        f"PaddleOCR-VL failed to process batch {batch_number}: {e}. "
                        "Check that the local VLM recognition server is reachable."
                    )
                )

            if len(predictions) != len(page_numbers):
                raise RequestException(
                    message=(
                        f"PaddleOCR-VL returned {len(predictions)} results for "
                        f"{len(page_numbers)} requested pages."
                    )
                )

            layouts = []
            for page_number, image, prediction in zip(page_numbers, page_images, predictions):
                layouts.append(
                    paddle_result_to_page_layout(
                        prediction,
                        page_number=page_number,
                        image_size=(image.width, image.height),
                        pdf_size=original_sizes.get(page_number, (image.width, image.height)),
                    )
                )

        print(
            f"PaddleOCR-VL batch {batch_number}: {len(layouts)} pages in {time.time() - start:.2f}s"
        )
        return layouts

    @function(
        image=paddle_ocr_vl_image,
        timeout=30 * 60,
        cpu=4,
        memory=PADDLE_OCR_VL_MEMORY_IN_GB,
        ephemeral_disk=30,
        gpu=PADDLE_OCR_VL_GPU_MODELS,
        retries=Retries(max_retries=2),
        min_containers=1,
        max_containers=1,
    )
    def run(self, parse_result: ParseResult) -> ParseResult:
        if not _cuda_is_available():
            raise RequestException(
                message=(
                    "ocr_model='paddle-ocr-vl' requires a CUDA-equipped worker, "
                    "but no usable GPU is available."
                )
            )

        print("Start OCR inference with PaddleOCR-VL")
        ctx: RequestContext = RequestContext.get()

        image_dimensions = ImageDimensions(
            min_pixels=3136,
            max_pixels=11289600,
            target_dpi=200,
            upgrade_image_dpi=True,
        )
        parse_result = self.run_batch_processing(ctx, parse_result, image_dimensions)
        return route_after_ocr(parse_result, log_prefix="PaddleOCRVLTask")
