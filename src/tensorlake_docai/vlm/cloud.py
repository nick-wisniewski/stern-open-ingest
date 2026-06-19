# SPDX-License-Identifier: Apache-2.0
"""Post-OCR VLM extraction retained for key-value regions."""

import asyncio
import os
import time
import uuid
from pathlib import Path
from typing import List

from PIL import Image
from tensorlake.applications import RequestContext, Retries, cls, function

from tensorlake_docai.extraction.key_value_extraction_utils import (
    run_element_key_value_extraction_and_modify_page_elements,
)
from tensorlake_docai.models.intermediate_objects import ParseResult
from tensorlake_docai.models.layout_objects import PageLayoutElement
from tensorlake_docai.ocr.utils import BatchProcessor
from tensorlake_docai.pipeline.api import PageFragmentType
from tensorlake_docai.pipeline.output_formatter import format_final_output
from tensorlake_docai.pipeline.simple_page_creator import ImageDimensions
from tensorlake_docai.vlm.workflow_images import vlm_extraction_image

PADDING = 4
MEMORY_IN_GB = 8

FORM_FRAGMENT_TYPES = [
    PageFragmentType.FORM,
    PageFragmentType.KEY_VALUE_REGION,
]

SECRETS = [
    "GEMINI_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
]


def crop_elements(
    page_layout,
    padding,
    page_image,
    element_types,
    scale_factor=1.0,
    save_images=False,
    output_dir="cropped_images",
    filename_prefix=None,
):
    import numpy as np

    cropped_images: List[Image.Image] = []
    page_elements: List[PageLayoutElement] = []

    if save_images:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        if filename_prefix is None:
            filename_prefix = f"page_{uuid.uuid4().hex[:8]}"

    element_counter = {}

    for page_element in page_layout.elements:
        if page_element.fragment_type not in element_types:
            continue

        x1, y1, x2, y2 = map(int, page_element.bbox)
        x1 = int(x1 * scale_factor)
        y1 = int(y1 * scale_factor)
        x2 = int(x2 * scale_factor)
        y2 = int(y2 * scale_factor)

        img_arr = np.array(page_image)
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(img_arr.shape[1], x2 + padding)
        y2 = min(img_arr.shape[0], y2 + padding)
        cropped_img = img_arr[y1:y2, x1:x2]

        if cropped_img.size == 0:
            print(
                f"Skipping element with empty crop region "
                f"(bbox={page_element.bbox}, scaled_region=({x1},{y1},{x2},{y2}), "
                f"image_shape={img_arr.shape})"
            )
            continue

        cropped_image = Image.fromarray(cropped_img)
        cropped_images.append(cropped_image)
        page_elements.append(page_element)

        if save_images:
            element_type = page_element.fragment_type.name.lower()
            element_counter[element_type] = element_counter.get(element_type, 0) + 1
            filename = f"{filename_prefix}_{element_type}_{element_counter[element_type]:02d}.png"
            filepath = Path(output_dir) / filename

            try:
                cropped_image.save(filepath, format="PNG")
                print(
                    f"Saved cropped {element_type} image: {filepath} "
                    f"(scale_factor: {scale_factor})"
                )
            except Exception as e:
                print(f"Error saving cropped image {filepath}: {e}")

    return cropped_images, page_elements


@cls()
class VLMExtractionTask(BatchProcessor):
    def __init__(self):
        BatchProcessor.__init__(self, memory_gb=MEMORY_IN_GB)

    async def _process_page_extraction(
        self,
        page_layout,
        page_image,
        element_types,
        element_types_to_check,
        extraction_func,
        scale_factor,
        parse_result,
    ):
        cropped_images, elements = crop_elements(
            page_layout=page_layout,
            padding=PADDING,
            page_image=page_image,
            element_types=element_types + element_types_to_check,
            scale_factor=scale_factor,
            save_images=parse_result.request.debug,
        )

        input_tokens, output_tokens = await extraction_func(
            cropped_images=cropped_images,
            page_elements=elements,
            element_types=element_types,
            element_types_to_check=element_types_to_check,
            parse_result=parse_result,
        )

        for img in cropped_images:
            try:
                img.close()
            except Exception:
                pass

        return input_tokens, output_tokens

    async def _run_key_value_extraction(
        self,
        parse_result,
        page_images_dict,
        element_types,
        element_types_to_check,
        scale_factor=1.0,
    ):
        start_time = time.time()
        tasks = []

        for page_num, page_image in page_images_dict.items():
            page_layout = next(
                (
                    layout
                    for layout in parse_result.document_layout.pages
                    if layout.page_number == page_num
                ),
                None,
            )
            if not page_layout:
                continue

            tasks.append(
                self._process_page_extraction(
                    page_layout,
                    page_image,
                    element_types,
                    element_types_to_check,
                    run_element_key_value_extraction_and_modify_page_elements,
                    scale_factor,
                    parse_result,
                )
            )

        if tasks:
            results = await asyncio.gather(*tasks)
            for in_tok, out_tok in results:
                self.key_value_extraction_input_tokens += in_tok
                self.key_value_extraction_output_tokens += out_tok

        print(f"Time taken for key-value extraction: {time.time() - start_time} seconds")

    def should_preserve_existing_pages(self) -> bool:
        """VLM tasks modify existing page layouts in place."""
        return True

    def get_processing_batches(
        self, parse_result: ParseResult, image_dimensions: ImageDimensions = ImageDimensions()
    ):
        """Override to keep using locally rendered images for VLM tasks."""
        for doc_pages in self.page_creator.get_images_generator(parse_result, image_dimensions):
            page_numbers = list(doc_pages.page_images.keys())
            payload = {
                "page_images": dict(doc_pages.page_images),
                "scale_factor": doc_pages.scale_factor,
            }
            yield BatchProcessor.ProcessingBatch(
                page_numbers=page_numbers,
                payload_kind="images",
                payload=payload,
                original_sizes=doc_pages.original_sizes,
            )

    async def process_batch(self, processing_batch, batch_number):
        ctx = RequestContext.get()

        payload = processing_batch.payload or {}
        page_images_dict = payload.get("page_images", {})
        scale_factor = payload.get("scale_factor", 1.0)

        await self._run_key_value_extraction(
            self._current_parse_result,
            page_images_dict,
            FORM_FRAGMENT_TYPES,
            [PageFragmentType.FIGURE, PageFragmentType.TABLE],
            scale_factor,
        )

        for img in list(page_images_dict.values()):
            try:
                img.close()
            except Exception:
                pass
        page_images_dict.clear()

        import gc

        gc.collect()

        ctx.progress.update(current=batch_number, total=batch_number + 1)
        print(f"Batch {batch_number} completed")
        return []

    @function(
        image=vlm_extraction_image,
        timeout=30 * 60,
        cpu=2,
        memory=MEMORY_IN_GB,
        ephemeral_disk=2,
        secrets=SECRETS,
        retries=Retries(max_retries=2),
        max_containers=200,
        min_containers=int(os.getenv("TENSORLAKE_MIN_CONTAINERS", "0")),
    )
    def run(self, parse_result: ParseResult) -> ParseResult:
        self.key_value_extraction_input_tokens = 0
        self.key_value_extraction_output_tokens = 0

        if parse_result.request.key_value_extraction:
            print("Using batch processing for key-value extraction...")
            self._current_parse_result = parse_result

            ctx: RequestContext = RequestContext.get()
            image_dimensions = ImageDimensions(target_dpi=200, upgrade_image_dpi=False)
            parse_result = self.run_batch_processing(ctx, parse_result, image_dimensions)

            if hasattr(self, "_current_parse_result"):
                delattr(self, "_current_parse_result")

        if parse_result.usage:
            parse_result.usage.summarization_input_tokens_used = (
                self.key_value_extraction_input_tokens
            )
            parse_result.usage.summarization_output_tokens_used = (
                self.key_value_extraction_output_tokens
            )
        else:
            from tensorlake_docai.pipeline.api import Usage

            parse_result.usage = Usage(
                pages_parsed=0,
                summarization_input_tokens_used=self.key_value_extraction_input_tokens,
                summarization_output_tokens_used=self.key_value_extraction_output_tokens,
            )

        print(
            "Total key-value extraction tokens used - "
            f"Input: {self.key_value_extraction_input_tokens}, "
            f"Output: {self.key_value_extraction_output_tokens}"
        )
        print("VLM_EXTRACTION -> OutputFormatter")
        return format_final_output(parse_result)
