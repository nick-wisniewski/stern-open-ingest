# SPDX-License-Identifier: Apache-2.0
"""
Key-value extraction utilities for rendering detected regions as Markdown.
"""

import asyncio
import json
from typing import List, Optional, Tuple

from PIL import Image

from tensorlake_docai.models.intermediate_objects import ParseResult
from tensorlake_docai.models.layout_objects import PageLayoutElement
from tensorlake_docai.pipeline.api import PageFragmentType
from tensorlake_docai.prompts.prompts import get_key_value_prompt_messages


def convert_key_value_json_to_markdown(content: str) -> str:
    """Convert key-value JSON content to Markdown."""
    try:
        if isinstance(content, str):
            data = json.loads(content)
        else:
            data = content

        md_lines = []

        if isinstance(data, dict):
            for k, v in data.items():
                md_lines.append(f"**{k}**: {v}")

        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    # Specific schema used by key-value extraction.
                    if "field_name" in item and "value" in item:
                        box_id = item.get("box_id", "")
                        name = item.get("field_name", "")
                        field_type = item.get("type", "")
                        val = item.get("value", "")

                        line = f"**{name}**"
                        if box_id:
                            line = f"[{box_id}] {line}"
                        if field_type:
                            line = f"{line} ({field_type})"
                        line = f"{line}: {val}"
                        md_lines.append(line)
                    else:
                        k = (
                            item.get("key")
                            or item.get("field")
                            or item.get("label")
                            or item.get("question")
                        )
                        v = (
                            item.get("value")
                            or item.get("text")
                            or item.get("answer")
                            or item.get("response")
                        )

                        if k is not None and v is not None:
                            md_lines.append(f"**{k}**: {v}")
                        elif len(item) == 1:
                            k, v = list(item.items())[0]
                            md_lines.append(f"**{k}**: {v}")
                        else:
                            for sub_k, sub_v in item.items():
                                md_lines.append(f"**{sub_k}**: {sub_v}")
                else:
                    md_lines.append(str(item))

        if md_lines:
            return "\n\n".join(md_lines)

        return content if isinstance(content, str) else json.dumps(content)

    except (json.JSONDecodeError, TypeError):
        return content if isinstance(content, str) else str(content)


async def run_element_key_value_extraction_and_modify_page_elements(
    cropped_images: List[Image.Image],
    page_elements: List[PageLayoutElement],
    element_types: List[PageFragmentType],
    element_types_to_check: List[PageFragmentType],
    parse_result: Optional[ParseResult] = None,
) -> Tuple[int, int]:

    total_input_tokens = 0
    total_output_tokens = 0

    tasks = []

    for img, element in zip(cropped_images, page_elements):
        tasks.append(_process_single_element(img, element, element_types, element_types_to_check))

    results = await asyncio.gather(*tasks)

    for input_tokens, output_tokens in results:
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens

    return total_input_tokens, total_output_tokens


async def _process_single_element(
    image: Image.Image,
    element: PageLayoutElement,
    target_types: List[PageFragmentType],
    check_types: List[PageFragmentType],
) -> Tuple[int, int]:
    from tensorlake_docai.providers.model_provider_utils import run_clients, _make_gemini_call

    input_tokens = 0
    output_tokens = 0

    is_key_value_region = False

    if element.fragment_type in target_types:
        is_key_value_region = True
    elif element.fragment_type in check_types:
        _, user_prompt = get_key_value_prompt_messages("detection")
        json_schema = json.dumps(
            {
                "type": "object",
                "properties": {"is_key_value_region": {"type": "boolean"}},
                "required": ["is_key_value_region"],
            }
        )

        try:
            response_text, in_tok, out_tok = await run_clients(
                user_prompt=user_prompt,
                images=[image],
                models=[_make_gemini_call],
                job_type="json_schema",
                json_schema=json_schema,
            )
            input_tokens += in_tok
            output_tokens += out_tok

            result = json.loads(response_text)
            if result.get("is_key_value_region", False):
                is_key_value_region = True
                element.fragment_type = PageFragmentType.FORM
        except Exception as e:
            print(f"Key-value region detection failed: {e}", flush=True)

    if is_key_value_region:
        _, user_prompt = get_key_value_prompt_messages("extraction")

        extraction_schema = json.dumps(
            {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "box_id": {
                            "type": "string",
                            "description": "The box id of the field, if present. Empty string if not available.",
                        },
                        "field_name": {
                            "type": "string",
                            "description": "The name or context of the field",
                        },
                        "type": {
                            "type": "string",
                            "description": "The field type, e.g. text, checkbox, radio button",
                        },
                        "value": {"type": "string", "description": "The field value"},
                    },
                    "required": ["field_name", "type", "value"],
                },
            }
        )

        try:
            response_text, in_tok, out_tok = await run_clients(
                user_prompt=user_prompt,
                images=[image],
                models=[_make_gemini_call],
                job_type="json_schema",
                json_schema=extraction_schema,
            )
            input_tokens += in_tok
            output_tokens += out_tok

            element.ocr_text = response_text
            element.html = ""
            element.markdown = convert_key_value_json_to_markdown(response_text)

        except Exception as e:
            print(f"Key-value extraction failed: {e}", flush=True)

    return input_tokens, output_tokens
