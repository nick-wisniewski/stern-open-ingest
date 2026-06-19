# SPDX-License-Identifier: Apache-2.0
# Document Understanding Tasks. This function batches the post-OCR VLM passes:
# 1. Figure and Table summarization
# 2. Page classification

import asyncio
import json
import time
import os
from tensorlake_docai.pipeline.api import (
    # "ANTHROPIC_API_KEY",
    PageFragmentType,
)
from tensorlake_docai.models.intermediate_objects import ParseResult

from tensorlake_docai.vlm.chart_extraction_utils import (
    run_element_chart_extraction_and_modify_page_elements,
)
from tensorlake_docai.vlm.element_summary_utils import (
    crop_elements,
    run_element_summary_and_modify_page_elements,
)
from tensorlake.applications import function, Retries, cls, RequestContext
from tensorlake_docai.vlm.workflow_images import vlm_extraction_image
from tensorlake_docai.pipeline.routing import (
    create_classification_choice_and_prompt,
)
from tensorlake_docai.pipeline.simple_page_creator import ImageDimensions
from tensorlake_docai.pipeline.output_formatter import format_final_output
from tensorlake_docai.extraction.key_value_extraction_utils import (
    run_element_key_value_extraction_and_modify_page_elements,
)
from tensorlake_docai.tables.table_cell_grounding_utils import (
    run_element_table_cell_grounding_and_modify_page_elements,
)
from tensorlake_docai.vlm.figure_grounding_utils import (
    run_element_figure_grounding_and_modify_page_elements,
)
from tensorlake_docai.ocr.utils import BatchProcessor

PADDING = 4
TABLE_WHITE_PADDING = 100

SECRETS = [
    "GEMINI_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
]

MEMORY_IN_GB = 8

TEXT_FRAGMENT_TYPES = [
    PageFragmentType.TITLE,
    PageFragmentType.SECTION_HEADER,
    PageFragmentType.TEXT,
    PageFragmentType.LIST_ITEM,
    PageFragmentType.TABLE_CAPTION,
    PageFragmentType.FIGURE_CAPTION,
    PageFragmentType.FORMULA_CAPTION,
    PageFragmentType.PAGE_FOOTER,
    PageFragmentType.PAGE_HEADER,
    PageFragmentType.PAGE_NUMBER,
]

TABLE_FRAGMENT_TYPES = [
    PageFragmentType.TABLE,
    PageFragmentType.DOCUMENT_INDEX,
]

FORM_FRAGMENT_TYPES = [
    PageFragmentType.FORM,
    PageFragmentType.KEY_VALUE_REGION,
]

FIGURE_FRAGMENT_TYPES = [
    PageFragmentType.FIGURE,
    PageFragmentType.FORMULA,
]


def clean_page_classification_result(raw_result):
    """
    Clean up page classification result to extract page class and reason.

    Handles various formats:
    - Direct class name: "form125"
    - JSON string: '{"page_class": "form125"}'
    - JSON with reason: '{"page_classes": ["form140"], "reason": "..."}'
    - Markdown wrapped JSON: '```json\n{"page_class": "form125"}\n```'
    - JSON array: '["form125", "form126"]' in multi_label classification

    Args:
        raw_result: Raw result from the model

    Returns:
        tuple: (page_class, reason, confidence) where reason and confidence can be None
    """
    if not raw_result:
        return "unknown", None, None

    # Convert to string if not already
    result_str = str(raw_result).strip()

    # Remove markdown code blocks if present
    if result_str.startswith("```"):
        # Find the content between code blocks
        lines = result_str.split("\n")
        content_lines = []
        in_code_block = False

        for line in lines:
            if line.strip().startswith("```"):
                in_code_block = not in_code_block
                continue
            if in_code_block:
                content_lines.append(line)

        result_str = "\n".join(content_lines).strip()

    # Try to parse as JSON
    try:
        parsed = json.loads(result_str)
        if isinstance(parsed, dict):
            reason = parsed.get("reason", None)
            confidence = parsed.get("confidence", None)
            # Clamp confidence to [0.0, 1.0] if present
            if confidence is not None:
                try:
                    confidence = max(0.0, min(1.0, float(confidence)))
                except (TypeError, ValueError):
                    confidence = None
            # Multi-label: return the list if present
            if "page_classes" in parsed:
                return parsed["page_classes"], reason, confidence
            # Multi-class: return the single class if present
            elif "page_class" in parsed:
                return parsed["page_class"], reason, confidence
            else:
                return "unknown", reason, confidence
        else:
            # If it's not a dict, use the parsed value directly
            return str(parsed), None, None
    except (json.JSONDecodeError, ValueError):
        # If JSON parsing fails, return the cleaned string
        return result_str, None, None


@cls()
class VLMExtractionTask(BatchProcessor):
    def __init__(self):
        # Don't pass batch_size=None explicitly - let BatchProcessor use default
        BatchProcessor.__init__(
            self, memory_gb=MEMORY_IN_GB
        )  # Let SimplePageCreator calculate batch size dynamically

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

        # Explicitly cleanup cropped images to free memory
        for img in cropped_images:
            try:
                img.close()
            except Exception:
                pass
        del cropped_images

        return input_tokens, output_tokens

    async def _run_extraction_common(
        self,
        parse_result,
        page_images_dict,
        element_types,
        element_types_to_check,
        extraction_func,
        task_name,
        scale_factor=1.0,
    ):
        start_time = time.time()
        total_input_tokens = 0
        total_output_tokens = 0

        tasks = []

        for page_num, page_image in page_images_dict.items():
            # Find the corresponding page layout
            page_layout = None
            for layout in parse_result.document_layout.pages:
                if layout.page_number == page_num:
                    page_layout = layout
                    break

            if not page_layout:
                continue

            tasks.append(
                self._process_page_extraction(
                    page_layout,
                    page_image,
                    element_types,
                    element_types_to_check,
                    extraction_func,
                    scale_factor,
                    parse_result,
                )
            )

        if tasks:
            results = await asyncio.gather(*tasks)
            for in_tok, out_tok in results:
                total_input_tokens += in_tok
                total_output_tokens += out_tok

        end_time = time.time()
        print(f"Time taken for {task_name}: {end_time - start_time} seconds")
        return total_input_tokens, total_output_tokens

    async def _run_chart_extraction(
        self,
        parse_result,
        page_images_dict,
        element_types,
        element_types_to_check,
        scale_factor=1.0,
    ):
        input_tokens, output_tokens = await self._run_extraction_common(
            parse_result,
            page_images_dict,
            element_types,
            element_types_to_check,
            run_element_chart_extraction_and_modify_page_elements,
            "chart extraction",
            scale_factor,
        )
        # Track chart extraction token usage
        self.chart_extraction_input_tokens += input_tokens
        self.chart_extraction_output_tokens += output_tokens

    async def _run_table_cell_grounding(
        self,
        parse_result,
        page_images_dict,
        element_types,
        element_types_to_check,
        scale_factor=1.0,
    ):
        input_tokens, output_tokens = await self._run_extraction_common(
            parse_result,
            page_images_dict,
            element_types,
            element_types_to_check,
            run_element_table_cell_grounding_and_modify_page_elements,
            "table cell grounding",
            scale_factor,
        )
        # Track table extraction token usage
        self.table_extraction_input_tokens += input_tokens
        self.table_extraction_output_tokens += output_tokens

    async def _run_figure_grounding(
        self, parse_result, page_images_dict, element_types, scale_factor=1.0
    ):
        input_tokens, output_tokens = await self._run_extraction_common(
            parse_result,
            page_images_dict,
            element_types,
            [],  # no element types to check, only figures
            run_element_figure_grounding_and_modify_page_elements,
            "figure grounding",
            scale_factor,
        )
        # Track figure grounding token usage
        if not hasattr(self, "figure_grounding_input_tokens"):
            self.figure_grounding_input_tokens = 0
            self.figure_grounding_output_tokens = 0
        self.figure_grounding_input_tokens += input_tokens
        self.figure_grounding_output_tokens += output_tokens

    async def _run_key_value_extraction(
        self,
        parse_result,
        page_images_dict,
        element_types,
        element_types_to_check,
        scale_factor=1.0,
    ):
        input_tokens, output_tokens = await self._run_extraction_common(
            parse_result,
            page_images_dict,
            element_types,
            element_types_to_check,
            run_element_key_value_extraction_and_modify_page_elements,
            "key-value extraction",
            scale_factor,
        )
        # Track key-value extraction token usage
        self.key_value_extraction_input_tokens += input_tokens
        self.key_value_extraction_output_tokens += output_tokens

    async def _summarize_single_page(
        self,
        page_layout,
        page_image,
        element_types,
        user_prompt,
        save_images,
        include_full_page_image,
        scale_factor,
    ):
        """
        Process summarization for a single page.
        Returns tuple of (input_tokens, output_tokens).
        """
        cropped_images, elements = crop_elements(
            page_layout=page_layout,
            padding=PADDING,
            page_image=page_image,
            element_types=element_types,
            scale_factor=scale_factor,
            save_images=save_images,
        )

        input_tokens, output_tokens = await run_element_summary_and_modify_page_elements(
            cropped_images=cropped_images,
            page_elements=elements,
            page_image=page_image if include_full_page_image else None,
            user_prompt=user_prompt,
            element_types=element_types,
        )

        # Explicitly cleanup cropped images to free memory
        for img in cropped_images:
            try:
                img.close()
            except Exception:
                pass
        del cropped_images

        return input_tokens, output_tokens

    async def _run_element_summarization(
        self,
        parse_result,
        page_images_dict,
        element_types,
        user_prompt,
        element_name,
        scale_factor=1.0,
    ):
        start_time = time.time()

        # Collect all tasks for concurrent processing
        summarization_tasks = []

        for page_num, page_image in page_images_dict.items():
            # Find the corresponding page layout
            page_layout = None
            for layout in parse_result.document_layout.pages:
                if layout.page_number == page_num:
                    page_layout = layout
                    break

            if not page_layout:
                continue

            # Create async task for this page
            summarization_tasks.append(
                self._summarize_single_page(
                    page_layout=page_layout,
                    page_image=page_image,
                    element_types=element_types,
                    user_prompt=user_prompt,
                    save_images=parse_result.request.debug,
                    include_full_page_image=parse_result.request.include_full_page_image,
                    scale_factor=scale_factor,
                )
            )

        # Process all pages concurrently in within a batch
        if summarization_tasks:
            print(
                f"Processing {len(summarization_tasks)} pages concurrently for {element_name} summarization..."
            )
            results_with_tokens = await asyncio.gather(*summarization_tasks)

            # Aggregate token usage
            for input_tokens, output_tokens in results_with_tokens:
                self.summarization_input_tokens += input_tokens
                self.summarization_output_tokens += output_tokens

        end_time = time.time()
        print(f"Time taken for {element_name} summary: {end_time - start_time} seconds")

    def should_preserve_existing_pages(self) -> bool:
        """
        VLM tasks modify existing page layouts in place rather than creating new ones.
        """
        return True

    def get_processing_batches(
        self, parse_result: ParseResult, image_dimensions: ImageDimensions = ImageDimensions()
    ):
        """
        Override to keep using locally rendered images for VLM tasks.
        """
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
        # Get context for progress tracking
        ctx = RequestContext.get()

        payload = processing_batch.payload or {}
        page_images_dict = payload.get("page_images", {})
        scale_factor = payload.get("scale_factor", 1.0)

        # Store the current request for processing
        current_request = self._current_parse_result.request

        # Process table summarization for this batch
        if current_request.table_summarization:
            await self._run_element_summarization(
                self._current_parse_result,
                page_images_dict,
                [PageFragmentType.TABLE],
                current_request.table_summarization_prompt,
                "table",
                scale_factor,
            )

        # Process figure summarization for this batch
        if current_request.figure_summarization:
            await self._run_element_summarization(
                self._current_parse_result,
                page_images_dict,
                [PageFragmentType.FIGURE],
                current_request.figure_summarization_prompt,
                "figure",
                scale_factor,
            )

        if current_request.chart_extraction:
            await self._run_chart_extraction(
                self._current_parse_result,
                page_images_dict,
                [PageFragmentType.CHART],
                [PageFragmentType.FIGURE, PageFragmentType.TABLE],
                scale_factor,
            )

        if current_request.key_value_extraction:
            await self._run_key_value_extraction(
                self._current_parse_result,
                page_images_dict,
                FORM_FRAGMENT_TYPES,
                [PageFragmentType.FIGURE, PageFragmentType.TABLE],
                scale_factor,
            )

        if current_request.table_cell_grounding:
            await self._run_table_cell_grounding(
                self._current_parse_result,
                page_images_dict,
                TABLE_FRAGMENT_TYPES + [PageFragmentType.TEXT],
                [],
                scale_factor,
            )

        if current_request.figure_grounding:
            await self._run_figure_grounding(
                self._current_parse_result,
                page_images_dict,
                [PageFragmentType.FIGURE],
                scale_factor,
            )

        # Process page classification for this batch
        if current_request.page_classification_request:
            await self._process_page_classification_batch(page_images_dict, batch_number)

        # Cleanup page images to free memory immediately
        for page_num, img in list(page_images_dict.items()):
            try:
                img.close()
            except Exception:
                pass
        page_images_dict.clear()

        # Force garbage collection to free memory
        import gc

        gc.collect()

        ctx.progress.update(current=batch_number, total=batch_number + 1)
        print(f"Batch {batch_number} completed: {len(page_images_dict)} pages processed")
        # Return empty list since we modify pages in-place (should_preserve_existing_pages=True)
        return []

    async def _process_page_classification_batch(self, page_images_dict, batch_number):
        """Process page classification for a batch of pages."""
        print(
            f"Running page classification for batch {batch_number}: {len(page_images_dict)} pages"
        )

        page_classification_start_time = time.time()

        # Get classification configuration
        page_classification_schema, _, page_classification_prompt = (
            create_classification_choice_and_prompt(
                self._current_parse_result.request.page_classification_request.class_definitions,
                self._current_parse_result.request.page_classification_request.classification_type,
            )
        )

        from tensorlake_docai.providers.model_provider_utils import (
            _make_gemini_call,
            _make_oai_call,
            run_clients,
        )

        page_classification_models = [_make_gemini_call, _make_oai_call]

        page_classification_tasks = []
        pages_for_classification = []

        for page_num, page_image in page_images_dict.items():
            # Find the corresponding page layout
            page_layout = None
            for layout in self._current_parse_result.document_layout.pages:
                if layout.page_number == page_num:
                    page_layout = layout
                    break

            if page_layout:
                pages_for_classification.append(page_layout)
                page_classification_tasks.append(
                    run_clients(
                        user_prompt=page_classification_prompt,
                        images=[page_image],
                        models=page_classification_models,
                        json_schema=page_classification_schema,
                        job_type="page_classification",
                    )
                )

        print(
            f"Batch {batch_number}: Awaiting {len(page_classification_tasks)} page classification tasks..."
        )
        results_with_tokens = await asyncio.gather(*page_classification_tasks)

        # Extract results and track tokens
        raw_results = []
        for result, input_tokens, output_tokens in results_with_tokens:
            raw_results.append(result)
            self.page_classification_input_tokens += input_tokens
            self.page_classification_output_tokens += output_tokens

        for page_layout, raw_result in zip(pages_for_classification, raw_results):
            page_class, reason, confidence = clean_page_classification_result(raw_result)
            page_layout.page_class = page_class
            page_layout.classification_reason = reason
            page_layout.classification_confidence = confidence
            # print(f"Batch {batch_number}: Page {page_layout.page_number} classified as: {page_class}")

        page_classification_end_time = time.time()
        print(
            f"Batch {batch_number} page classification time: {page_classification_end_time - page_classification_start_time:.2f} seconds"
        )

    @function(
        image=vlm_extraction_image,
        timeout=30 * 60,  # 30 minutes
        cpu=2,
        memory=MEMORY_IN_GB,
        # The function is not using /tmp disk space, just reserve a small amount
        ephemeral_disk=2,
        secrets=SECRETS,
        retries=Retries(max_retries=2),
        max_containers=200,
        min_containers=int(os.getenv("TENSORLAKE_MIN_CONTAINERS", "0")),
    )
    def run(self, parse_result: ParseResult) -> ParseResult:
        # Initialize token tracking at the start of each run
        self.summarization_input_tokens = 0
        self.summarization_output_tokens = 0
        self.page_classification_input_tokens = 0
        self.page_classification_output_tokens = 0
        self.chart_extraction_input_tokens = 0
        self.chart_extraction_output_tokens = 0
        self.table_extraction_input_tokens = 0
        self.table_extraction_output_tokens = 0
        self.key_value_extraction_input_tokens = 0
        self.key_value_extraction_output_tokens = 0
        self.figure_grounding_input_tokens = 0
        self.figure_grounding_output_tokens = 0

        # Determine if any VLM task requires page images.
        needs_images = (
            parse_result.request.table_summarization
            or parse_result.request.figure_summarization
            or parse_result.request.chart_extraction
            or parse_result.request.table_cell_grounding
            or parse_result.request.key_value_extraction
            or parse_result.request.figure_grounding
            or parse_result.request.page_classification_request
        )

        if needs_images:
            print("🔧 Using batch processing for VLM tasks...")

            # Store current parse result for batch processing methods to access
            self._current_parse_result = parse_result

            # Run batch processing - this will process each batch through process_batch method
            ctx: RequestContext = RequestContext.get()
            image_dimensions = ImageDimensions(target_dpi=200, upgrade_image_dpi=False)
            parse_result = self.run_batch_processing(ctx, parse_result, image_dimensions)

            print("✅ Completed VLM batch processing")

            # Clean up current parse result reference
            if hasattr(self, "_current_parse_result"):
                delattr(self, "_current_parse_result")

        # Update usage with VLM token information.
        if parse_result.usage:
            # Set summarization token usage (summarization only happens here)
            parse_result.usage.summarization_input_tokens_used = (
                self.summarization_input_tokens
                + self.chart_extraction_input_tokens
                + self.key_value_extraction_input_tokens
                + self.table_extraction_input_tokens
                + self.figure_grounding_input_tokens
                + self.page_classification_input_tokens
            )
            parse_result.usage.summarization_output_tokens_used = (
                self.summarization_output_tokens
                + self.chart_extraction_output_tokens
                + self.key_value_extraction_output_tokens
                + self.table_extraction_output_tokens
                + self.figure_grounding_output_tokens
                + self.page_classification_output_tokens
            )
        else:
            # Create new usage object with VLM token counts.
            from tensorlake_docai.pipeline.api import Usage

            parse_result.usage = Usage(
                pages_parsed=0,  # Will be set by other workflow components
                summarization_input_tokens_used=self.summarization_input_tokens
                + self.chart_extraction_input_tokens
                + self.key_value_extraction_input_tokens
                + self.table_extraction_input_tokens
                + self.figure_grounding_input_tokens
                + self.page_classification_input_tokens,
                summarization_output_tokens_used=self.summarization_output_tokens
                + self.chart_extraction_output_tokens
                + self.key_value_extraction_output_tokens
                + self.table_extraction_output_tokens
                + self.figure_grounding_output_tokens
                + self.page_classification_output_tokens,
            )

        print(
            f"Total summarization tokens used - Input: {self.summarization_input_tokens}, Output: {self.summarization_output_tokens}"
        )
        print("🔀 VLM_EXTRACTION → OutputFormatter")
        return format_final_output(parse_result)
