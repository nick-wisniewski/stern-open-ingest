# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import json
import math
import os
from dataclasses import dataclass, asdict
from typing import Dict


from PIL import Image
import base64
import tempfile
from tensorlake.applications import cls, function
from tensorlake_docai.vlm.workflow_images import file_convertion_image
from tensorlake_docai.pipeline.api import Usage, ParsedDocumentRef
from tensorlake_docai.models.intermediate_objects import ParseResult, FormFillingResult
from tensorlake_docai.pipeline.output_formatter import format_final_output

# --- Data Models ---


@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def width(self) -> float:
        return abs(self.x2 - self.x1)

    @property
    def height(self) -> float:
        return abs(self.y2 - self.y1)

    def distance_to(self, other: BoundingBox) -> float:
        cx1, cy1 = self.center
        cx2, cy2 = other.center
        return math.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)

    def iou(self, other: BoundingBox) -> float:
        """Calculates Intersection over Union (IoU) with another box."""
        x_left = max(self.x1, other.x1)
        y_top = max(self.y1, other.y1)
        x_right = min(self.x2, other.x2)
        y_bottom = min(self.y2, other.y2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        area1 = self.width * self.height
        area2 = other.width * other.height

        union_area = area1 + area2 - intersection_area
        if union_area == 0:
            return 0.0

        return intersection_area / union_area

    def is_contained_in(self, other: BoundingBox, threshold: float = 0.9) -> bool:
        """Checks if this box is mostly contained within another box."""
        x_left = max(self.x1, other.x1)
        y_top = max(self.y1, other.y1)
        x_right = min(self.x2, other.x2)
        y_bottom = min(self.y2, other.y2)

        if x_right < x_left or y_bottom < y_top:
            return False

        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        self_area = self.width * self.height
        if self_area == 0:
            return False
        return (intersection_area / self_area) >= threshold


@dataclass
class PageFragment:
    """Represents a text element or figure from the PDF parsing stage."""

    fragment_type: str
    content: str
    bbox: BoundingBox
    reading_order: int


@dataclass
class DetectedWidget:
    """Represents a form widget found by the object detector."""

    label: str  # e.g., 'text_input', 'checkbox'
    score: float
    bbox: BoundingBox
    linked_text: str | None = None  # The predicted label/question for this widget
    is_filled: bool = False  # Indicates if the detector thinks this is already filled
    page_number: int = 1
    # Indicates if the widget was already present in the source PDF
    is_existing: bool = False
    description: str | None = None
    text_content: str | None = None
    field_name: str | None = None
    # Context for disambiguation
    surrounding_text: str | None = None


@dataclass
class PageData:
    page_number: int
    fragments: list[PageFragment]
    # In a real scenario, this would hold the image reference for the detector
    image_path: str | None = None


# --- Core Logic Components ---


class LabelAssociator:
    """
    Responsible for linking detected widgets to their semantic text labels
    based on geometric proximity and alignment.
    """

    def associate(
        self,
        widget: DetectedWidget,
        fragments: list[PageFragment],
    ) -> str | None:
        """
        Finds the text fragment most likely associated with the widget.
        Prioritizes text to the Left (key-value pairs) or Above (headers).
        """
        best_candidate = None
        min_score = float("inf")

        w_x1, w_y1, w_x2, w_y2 = widget.bbox.x1, widget.bbox.y1, widget.bbox.x2, widget.bbox.y2
        w_center_y = (w_y1 + w_y2) / 2

        y_min_below = -20
        y_max_above = 100
        top_penalty = 500

        for frag in fragments:
            if frag.fragment_type not in ["text", "title", "section_header"]:
                continue

            f_x1, f_y1, f_x2, f_y2 = frag.bbox.x1, frag.bbox.y1, frag.bbox.x2, frag.bbox.y2
            f_center_y = (f_y1 + f_y2) / 2

            # Vertical alignment check
            y_diff = w_center_y - f_center_y

            if y_diff < y_min_below:
                continue  # Text is significantly below
            if y_diff > y_max_above:
                continue  # Text is significantly above

            # Horizontal relationship
            left_gap = w_x1 - f_x2  # Positive if text is to left
            right_gap = f_x1 - w_x2  # Positive if text is to right

            score = float("inf")

            # 1. Text is to the Left (Label: [Input])
            if left_gap >= -10:
                dist_score = left_gap if left_gap >= 0 else abs(left_gap) * 2
                v_penalty = abs(y_diff) * 2
                score = dist_score + v_penalty

            # 2. Text is Above (Label\n[Input])
            elif f_y2 <= w_y1 + 10:
                w_center_x = (w_x1 + w_x2) / 2
                f_center_x = (f_x1 + f_x2) / 2
                h_misalignment = abs(w_center_x - f_center_x)
                vertical_gap = w_y1 - f_y2
                score = vertical_gap + h_misalignment + top_penalty  # Prefer Left over Top

            # 3. Text is to the Right ([ ] Label)
            elif right_gap >= -10:
                dist_score = right_gap if right_gap >= 0 else abs(right_gap) * 2
                v_penalty = abs(y_diff) * 2
                if widget.label in ["checkbox", "radio"]:
                    score = dist_score + v_penalty + 50
                else:
                    score = (
                        dist_score + v_penalty + 1000
                    )  # Strong penalty for right-side text on inputs

            if score < min_score:
                min_score = score
                best_candidate = frag

        return best_candidate.content if best_candidate else None


class PdfFormAugmenter:
    """
    Handles the modification of the PDF to add interactive fields
    and generates debug information.
    """

    def augment_pdf(
        self,
        source_pdf_path: str,
        output_pdf_path: str,
        widgets: list[DetectedWidget],
    ):
        """
        Creates a new PDF with the additional widgets using pypdf.
        """
        import pypdf

        try:
            from pypdf.generic import (
                NameObject,
                DictionaryObject,
                ArrayObject,
                FloatObject,
                TextStringObject,
                NumberObject,
                BooleanObject,
            )

            reader = pypdf.PdfReader(source_pdf_path)
            writer = pypdf.PdfWriter()
            # Use clone_document_from_reader to properly copy the entire document structure,
            # including the AcroForm and its resources, which append_pages_from_reader does not.
            writer.clone_document_from_reader(reader)

            # Ensure we have an AcroForm dictionary to register fields
            if "/AcroForm" not in writer.root_object:
                writer.root_object[NameObject("/AcroForm")] = DictionaryObject()

            acroform = writer.root_object["/AcroForm"]

            # NeedAppearances is crucial for seeing the widgets without clicking them
            acroform[NameObject("/NeedAppearances")] = BooleanObject(True)

            # Ensure Default Resources (DR) are present for fonts used in DA strings
            if "/DR" not in acroform:
                acroform[NameObject("/DR")] = DictionaryObject()

            # Ensure Default Appearance (DA) is present globally
            if "/DA" not in acroform:
                acroform[NameObject("/DA")] = TextStringObject("/Helv 0 Tf 0 g")

            dr = acroform["/DR"]
            if "/Font" not in dr:
                dr[NameObject("/Font")] = DictionaryObject()

            font_dict = dr["/Font"]
            if "/Helv" not in font_dict:
                helv_font = DictionaryObject()
                helv_font[NameObject("/Type")] = NameObject("/Font")
                helv_font[NameObject("/Subtype")] = NameObject("/Type1")
                helv_font[NameObject("/BaseFont")] = NameObject("/Helvetica")
                font_dict[NameObject("/Helv")] = helv_font

            if "/Fields" not in acroform:
                acroform[NameObject("/Fields")] = ArrayObject()

            fields = acroform["/Fields"]

            for w in widgets:
                page_idx = w.page_number - 1
                if page_idx < 0 or page_idx >= len(writer.pages):
                    continue

                page = writer.pages[page_idx]
                mb = page.mediabox

                if w.is_existing:
                    if (w.is_filled or w.text_content) and "/Annots" in page:
                        page_top = float(mb.top)
                        page_left = float(mb.left)

                        for annot in page["/Annots"]:
                            obj = annot.get_object()
                            if obj.get("/Subtype") == "/Widget":
                                rect = obj.get("/Rect")
                                if not rect:
                                    continue
                                x_ll, y_ll, x_ur, y_ur = [float(c) for c in rect]
                                b_x1 = x_ll - page_left
                                b_y1 = page_top - y_ur

                                if abs(b_x1 - w.bbox.x1) < 1.0 and abs(b_y1 - w.bbox.y1) < 1.0:
                                    if w.label in ["checkbox", "radio"]:
                                        val = (
                                            NameObject("/Yes")
                                            if w.is_filled
                                            else NameObject("/Off")
                                        )
                                        obj[NameObject("/V")] = val
                                        obj[NameObject("/AS")] = val
                                    elif w.text_content:
                                        obj[NameObject("/V")] = TextStringObject(w.text_content)
                                        # Remove Appearance dictionary to force regeneration by the viewer
                                        if "/AP" in obj:
                                            del obj["/AP"]
                                    break
                    continue

                page_left = float(mb.left)
                page_bottom = float(mb.bottom)
                pg_w = float(mb.width)
                pg_h = float(mb.height)

                # Get rotation safely handling inheritance
                rotation = 0
                if hasattr(page, "rotation"):
                    rotation = page.rotation
                else:
                    rotation = page.get("/Rotate", 0)
                rotation = int(rotation) % 360

                # Transform coordinates from Image Space (Top-Left) to PDF Space (Bottom-Left, Unrotated)
                x1, y1, x2, y2 = w.bbox.x1, w.bbox.y1, w.bbox.x2, w.bbox.y2

                """
                if w.label == "text_input":
                    # Heuristic: Shrink the box slightly to prevent obscuring adjacent labels
                    # especially the label above the field.
                    w_box = abs(x2 - x1)
                    h_box = abs(y2 - y1)
                    x1 += w_box * 0.01
                    x2 -= w_box * 0.01
                    y1 += h_box * 0.10  # Shrink top by 10% (Image space y increases downwards)
                    y2 -= h_box * 0.02
                """
                bx_min, bx_max = min(x1, x2), max(x1, x2)
                by_min, by_max = min(y1, y2), max(y1, y2)

                # Ensure minimum dimensions for visibility
                if (bx_max - bx_min) < 10:
                    bx_max = bx_min + 10
                if (by_max - by_min) < 10:
                    by_max = by_min + 10

                if rotation == 0:
                    rect = [bx_min, pg_h - by_max, bx_max, pg_h - by_min]
                elif rotation == 90:
                    # Visual (0,0) -> PDF (0, 0)
                    rect = [by_min, bx_min, by_max, bx_max]
                elif rotation == 180:
                    # Visual (0,0) -> PDF (W, 0)
                    rect = [pg_w - bx_max, by_min, pg_w - bx_min, by_max]
                elif rotation == 270:
                    # Visual (0,0) -> PDF (W, H)
                    rect = [pg_w - by_max, pg_h - bx_max, pg_w - by_min, pg_h - bx_min]
                else:
                    rect = [bx_min, pg_h - by_max, bx_max, pg_h - by_min]

                # Apply offset
                rect = [
                    page_left + rect[0],
                    page_bottom + rect[1],
                    page_left + rect[2],
                    page_bottom + rect[3],
                ]

                # Create Widget

                # Create Annotation
                annot = DictionaryObject()
                annot[NameObject("/Type")] = NameObject("/Annot")
                annot[NameObject("/Subtype")] = NameObject("/Widget")
                annot[NameObject("/Rect")] = ArrayObject([FloatObject(c) for c in rect])

                annot[NameObject("/T")] = TextStringObject(w.field_name)
                annot[NameObject("/F")] = NumberObject(4)  # Print flag

                # Border Style
                bs = DictionaryObject()
                bs[NameObject("/W")] = NumberObject(1)
                bs[NameObject("/S")] = NameObject("/S")
                annot[NameObject("/BS")] = bs

                # Appearance Characteristics (Black Border)
                mk = DictionaryObject()
                mk[NameObject("/BC")] = ArrayObject(
                    [FloatObject(0), FloatObject(0), FloatObject(0)],
                )
                mk[NameObject("/BG")] = ArrayObject(
                    [FloatObject(0.9), FloatObject(0.95), FloatObject(1.0)],
                )

                mk[NameObject("/R")] = NumberObject(rotation)

                # Set MK dictionary
                annot[NameObject("/MK")] = mk

                if w.label in ["checkbox", "radio"]:
                    annot[NameObject("/FT")] = NameObject("/Btn")
                    if w.label == "radio":
                        annot[NameObject("/Ff")] = NumberObject(32768)
                    else:
                        annot[NameObject("/Ff")] = NumberObject(0)

                    if w.is_filled:
                        annot[NameObject("/V")] = NameObject("/Yes")
                        annot[NameObject("/AS")] = NameObject("/Yes")
                    else:
                        annot[NameObject("/V")] = NameObject("/Off")
                        annot[NameObject("/AS")] = NameObject("/Off")
                elif w.label == "signature":
                    annot[NameObject("/FT")] = NameObject("/Sig")
                else:
                    annot[NameObject("/FT")] = NameObject("/Tx")
                    annot[NameObject("/Ff")] = NumberObject(4096)  # Multiline
                    # Use 0 for font size to enable auto-sizing to fit the box
                    annot[NameObject("/DA")] = TextStringObject("/Helv 0 Tf 0 g")
                    if w.text_content:
                        annot[NameObject("/V")] = TextStringObject(w.text_content)

                # Link to the page to prevent "hanging" in some readers
                if hasattr(page, "indirect_ref"):
                    annot[NameObject("/P")] = page.indirect_ref
                elif hasattr(page, "indirectRef"):
                    annot[NameObject("/P")] = page.indirectRef

                # Use add_annotation to register the object and add to page
                annot_ref = writer.add_annotation(page_idx, annot)
                if annot_ref is None:
                    annot_ref = page["/Annots"][-1]

                # Also add to the global Fields array so it functions as a form
                fields.append(annot_ref)

            with open(output_pdf_path, "wb") as f:
                writer.write(f)
            print(f"Augmented PDF saved to {output_pdf_path}")

        except Exception as e:
            print(f"Error augmenting PDF: {e}")
            import traceback

            traceback.print_exc()


FORM_FILLING_PROMPTS = {
    "detection": {
        "system": "You are an expert form parser. Your task is to analyze a document page image and identify all form widgets present. Images dimensions should be mapped to 1000x1000 coordinate space, with (0,0) at the top-left. ",
        "user": "Inspect the provided image of a document page and identify and enumerate all form widgets present, use visual cues to identify all the areas where data could be input or has been entered. "
        "Ground the bounding boxes of the detected widgets to the image using the provided coordinates. "
        "The detected widgets will be overlayed over the form image. "
        "Return a JSON object with a key 'widgets' containing a list of detected widgets.\n"
        "Each item must have:\n"
        "'label' (one of: 'text_input', 'checkbox', 'radio', 'signature'),\n"
        "'box_2d' (bounding box in [ymin, xmin, ymax, xmax] format, normalized 0-1000 for both dimensions),\n"
        "'is_filled' (boolean),\n"
        "'text_content' (string, the text value inside the widget if present),\n"
        "'description' (short text describing the widget context).\n\n"
        "Identify if the widget has a label (e.g. agency, code, ...) and, revise the box_2d to ensure that the label is not included in the bounding box, to prevent that it is not visible when filling the form, but none of the detected widgets should be discarded because of this requirements, it should be possible to have widgets without a label nearby.",
    },
    "filling": {
        "system": (
            "You are an AI assistant that fills forms based on provided information. "
            "You will receive a list of fields and a text containing data. "
            "Map the data to the fields. Use semantic reasoning to match data to field labels even if they are not identical. "
            "For 'text_input', provide the string value. "
            "For 'checkbox' or 'radio', provide a boolean (true for checked/yes, false for unchecked/no). "
            "Return a JSON object where keys are the field 'id' and values are the assigned values."
        ),
        "user": "Data:\n{prompt_text}\n\nFields:\n{fields}",
    },
    "analyzer": {
        "system": (
            "You are a form analyzer. I will provide an image of a form where the widgets to analyze are highlighted with red bounding boxes and labeled with their IDs in red. "
            "I will also provide a list of these widgets with their IDs and bounding box coordinates (normalized 0-1000). "
            "The form image dimensions should be within a 1000x1000 coordinate space, so the bounding box coordinates correspond to positions on the image. "
            "You will ground the bounding boxes to the form image using both the visual red boxes/IDs and the coordinates. "
            "For each field identified by its ID, identify the semantic label (e.g., the text 'Name:' next to the box) "
            "and provide a short description of what should be entered. "
            "Also extract the surrounding text context (sentences, headers, or paragraphs containing this field) to help locate it in the document text. "
            "Use visual cues to target the specific label for each widget. "
            "Do not use text that belongs to other nearby widgets. "
            "Return a JSON object with a key 'fields' containing a list of objects: {'id': int, 'label': string, 'description': string, 'surrounding_text': string}."
        ),
        "user": "Widgets to analyze:\n{widgets}",
    },
}

FORM_FILLING_SCHEMAS = {
    "detection": {
        "type": "object",
        "properties": {
            "widgets": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {
                            "type": "string",
                            "enum": ["text_input", "checkbox", "radio", "signature"],
                        },
                        "box_2d": {
                            "type": "array",
                            "items": {"type": "integer", "minimum": 0, "maximum": 1000},
                            "minItems": 4,
                            "maxItems": 4,
                        },
                        "is_filled": {"type": "boolean"},
                        "text_content": {"type": "string"},
                        "description": {"type": "string"},
                    },
                    "required": ["label", "box_2d", "is_filled"],
                },
            }
        },
        "required": ["widgets"],
    },
    "analyzer": {
        "type": "object",
        "properties": {
            "fields": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "label": {"type": "string"},
                        "description": {"type": "string"},
                        "surrounding_text": {"type": "string"},
                    },
                    "required": ["id", "label", "description"],
                },
            }
        },
        "required": ["fields"],
    },
}


def get_form_filling_prompts(task: str) -> tuple[str, str]:
    try:
        prompt_data = FORM_FILLING_PROMPTS[task]
    except KeyError:
        raise ValueError(f"Invalid form filling task: {task}")
    return prompt_data["system"], prompt_data["user"]


def get_form_filling_schema(task: str) -> Dict:
    try:
        return FORM_FILLING_SCHEMAS[task]
    except KeyError:
        raise ValueError(f"Invalid form filling schema task: {task}")


class FormDetector:
    """
    Wrapper for Gemini and local models to detect form widgets in images.
    """

    def __init__(
        self,
        gemini_model_name: str = "gemini-3-flash-preview",
    ):
        self.gemini_model_name = gemini_model_name

    def detect_gemini(
        self,
        page_image_path: str,
        page_width: float,
        page_height: float,
    ) -> tuple[list[DetectedWidget], int, int]:
        """
        Runs Gemini detection.
        """
        from google.genai import types
        from tensorlake_docai.providers.model_provider_utils import _make_gemini_call

        input_tokens = 0
        output_tokens = 0
        if not os.path.exists(page_image_path):
            print(
                f"Skipping Gemini detection (Path: {page_image_path})",
            )
            return [], 0, 0

        print(f"Running Gemini detection on {page_image_path}...")

        system_prompt, user_prompt = get_form_filling_prompts("detection")

        detection_schema = get_form_filling_schema("detection")

        try:
            with Image.open(page_image_path) as img:
                config_overrides = {
                    "thinking_config": types.ThinkingConfig(thinking_level="high"),
                    "temperature": 0.0,
                }

                response_text, in_tok, out_tok = asyncio.run(
                    _make_gemini_call(
                        system_instruction=system_prompt,
                        user_prompt=user_prompt,
                        timeout=360,
                        images=[img],
                        model_name=self.gemini_model_name,
                        job_type="structured_extraction",
                        config_overrides=config_overrides,
                        json_schema=json.dumps(detection_schema),
                    )
                )
                input_tokens += in_tok
                output_tokens += out_tok
            data = json.loads(response_text)

            if isinstance(data, list):
                data = data[0] if data else {}

            widgets = []
            for w in data.get("widgets", []):
                box = w["box_2d"]
                # Convert [ymin, xmin, ymax, xmax] to [x1, y1, x2, y2] in page points.
                y1 = (box[0] / 1000) * page_height
                x1 = (box[1] / 1000) * page_width
                y2 = (box[2] / 1000) * page_height
                x2 = (box[3] / 1000) * page_width

                label = w.get("label", "text_input")

                # Heuristic: Expand checkboxes slightly as they are often detected too tightly
                if label in ["checkbox", "radio"]:
                    w_box = x2 - x1
                    x1 -= w_box * 0.1
                    x2 += w_box * 0.1

                text_content = w.get("text_content")
                if isinstance(text_content, str) and text_content.lower() == "null":
                    text_content = ""

                widgets.append(
                    DetectedWidget(
                        label=label,
                        score=1.0,
                        bbox=BoundingBox(x1, y1, x2, y2),
                        is_filled=w.get("is_filled", False),
                        description=w.get("description"),
                        text_content=text_content or "",
                    ),
                )

            return widgets, input_tokens, output_tokens
        except Exception as e:
            print(f"Gemini detection failed: {e}")
            return [], 0, 0


class FormFiller:
    """
    Uses an LLM to map a user prompt (data) to the detected form widgets.
    """

    def __init__(self, model_name: str = "gemini-3-flash-preview"):
        self.model_name = model_name

    def fill(self, widgets: list[DetectedWidget], prompt_text: str) -> tuple[int, int]:
        """
        Fills the widgets based on the provided prompt text.
        Updates the 'text_content' or 'is_filled' attributes of the widgets.
        """
        from tensorlake_docai.providers.model_provider_utils import _make_gemini_call

        input_tokens = 0
        output_tokens = 0
        if not widgets:
            return 0, 0

        properties = {}
        fields_desc = []
        for w in widgets:
            fields_desc.append(
                {
                    "id": w.field_name,
                    "label": w.linked_text or w.label,
                    "type": w.label,
                    "description": w.description,
                }
            )

            if w.label in ["checkbox", "radio"]:
                prop_type = "boolean"
            else:
                prop_type = "string"

            if w.field_name:
                properties[w.field_name] = {"type": prop_type}

        system_prompt, user_prompt = get_form_filling_prompts("filling")

        user_prompt = user_prompt.format(
            prompt_text=prompt_text, fields=json.dumps(fields_desc, indent=2)
        )

        try:
            response_text, in_tok, out_tok = asyncio.run(
                _make_gemini_call(
                    user_prompt=user_prompt,
                    images=[],
                    model_name=self.model_name,
                    system_instruction=system_prompt,
                    job_type="structured_extraction",
                    config_overrides={"temperature": 0.0},
                    # json_schema=json.dumps(filling_schema),
                )
            )
            input_tokens += in_tok
            output_tokens += out_tok

            result = json.loads(response_text)
            print(f"  LLM Response for filling: {json.dumps(result, indent=2)}")

            for w in widgets:
                if w.field_name in result:
                    val = result[w.field_name]
                    print(f"Filling field '{w.field_name}' with value: {val}")
                    if w.label in ["checkbox", "radio"]:
                        if isinstance(val, bool):
                            w.is_filled = val
                        elif isinstance(val, str):
                            w.is_filled = val.lower().strip() in [
                                "true",
                                "yes",
                                "checked",
                                "x",
                                "on",
                                "selected",
                            ]
                    else:
                        if val is not None:
                            str_val = str(val)
                            if str_val.lower() == "null":
                                w.text_content = ""
                            else:
                                w.text_content = str_val

        except Exception as e:
            print(f"Error during form filling: {e}")
        return input_tokens, output_tokens


class MetadataRefiner:
    """
    Uses a VLM to analyze widgets that lack proper labels (e.g. existing PDF fields
    with generic names) and assigns them semantic labels based on visual context.
    """

    def __init__(self, model_name: str = "gemini-3-flash-preview"):
        self.model_name = model_name

    def refine(
        self,
        widgets: list[DetectedWidget],
        image_path: str,
        width: float,
        height: float,
        extract_context: bool = False,
    ) -> tuple[int, int]:
        from tensorlake_docai.providers.model_provider_utils import _make_gemini_call
        from google.genai import types

        input_tokens = 0
        output_tokens = 0
        if not widgets:
            return 0, 0

        if not image_path or not os.path.exists(image_path):
            print(
                f"  Warning: Skipping metadata refinement for {len(widgets)} widgets due to missing image."
            )
            return 0, 0

        # Identify widgets that need refinement (missing linked_text or generic names)
        widgets_to_refine = []
        for i, w in enumerate(widgets):
            # Refine if no linked text, or if the field name looks like a generic ID
            if (
                extract_context
                or not w.linked_text
                or (w.field_name and "Check Box" in w.field_name)
            ):
                widgets_to_refine.append((i, w))

        if not widgets_to_refine:
            return 0, 0

        print(f"  Refining metadata for {len(widgets_to_refine)} widgets using VLM...")

        for i, w in widgets_to_refine:
            print(
                f"    - Queuing widget for refinement (id: {i}, name: '{w.field_name}', label: '{w.linked_text}')"
            )

        widget_inputs = []
        for idx, w in widgets_to_refine:
            # Normalize coordinates to 0-1000
            ymin = int((w.bbox.y1 / height) * 1000)
            xmin = int((w.bbox.x1 / width) * 1000)
            ymax = int((w.bbox.y2 / height) * 1000)
            xmax = int((w.bbox.x2 / width) * 1000)
            widget_inputs.append({"id": idx, "box_2d": [ymin, xmin, ymax, xmax]})

        analyzer_schema = get_form_filling_schema("analyzer")

        system_prompt, user_prompt = get_form_filling_prompts("analyzer")

        try:
            with Image.open(image_path) as img:
                # Draw bounding boxes and IDs on the image
                from PIL import ImageDraw, ImageFont

                draw_img = img.copy()
                draw = ImageDraw.Draw(draw_img)

                try:
                    try:
                        font = ImageFont.truetype("DejaVuSans.ttf", 20)
                    except OSError:
                        try:
                            font = ImageFont.load_default(size=20)
                        except TypeError:
                            font = ImageFont.load_default()
                except Exception:
                    font = None

                img_w, img_h = draw_img.size
                scale_x = img_w / width if width > 0 else 1.0
                scale_y = img_h / height if height > 0 else 1.0

                for idx, w in widgets_to_refine:
                    x1 = w.bbox.x1 * scale_x
                    y1 = w.bbox.y1 * scale_y
                    x2 = w.bbox.x2 * scale_x
                    y2 = w.bbox.y2 * scale_y

                    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

                    text = str(idx)
                    text_x, text_y = x1, y1
                    if font:
                        try:
                            left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
                            text_w = right - left
                            text_h = bottom - top
                            text_x = x1 + (x2 - x1 - text_w) / 2
                            text_y = y1 + (y2 - y1 - text_h) / 2
                        except AttributeError:
                            pass
                        draw.text((text_x, text_y), text, fill="red", font=font)
                    else:
                        draw.text((x1, y1), text, fill="red")

                config_overrides = {
                    "thinking_config": types.ThinkingConfig(thinking_level="high"),
                    "temperature": 0.0,
                }

                user_prompt = user_prompt.format(widgets=json.dumps(widget_inputs))
                response_text, in_tok, out_tok = asyncio.run(
                    _make_gemini_call(
                        system_instruction=system_prompt,
                        user_prompt=user_prompt,
                        images=[draw_img],
                        model_name=self.model_name,
                        job_type="structured_extraction",
                        config_overrides=config_overrides,
                        json_schema=json.dumps(analyzer_schema),
                    )
                )
                input_tokens += in_tok
                output_tokens += out_tok
            result = json.loads(response_text)

            for item in result.get("fields", []):
                idx = item.get("id")
                if idx is not None and 0 <= idx < len(widgets):
                    w = widgets[idx]
                    if item.get("label") and item["label"] != w.linked_text:
                        print(
                            f"    - Refining widget {idx}: setting label to '{item['label']}' (was '{w.linked_text}')"
                        )
                        w.linked_text = item["label"]
                    if item.get("description") and item["description"] != w.description:
                        print(
                            f"    - Refining widget {idx}: setting description to '{item['description']}' (was '{w.description}')"
                        )
                        w.description = item["description"]
                    if item.get("surrounding_text"):
                        w.surrounding_text = item["surrounding_text"]

        except Exception as e:
            print(f"  Metadata refinement failed: {e}")
        return input_tokens, output_tokens


class FormFillingSystem:
    def __init__(self):
        self.detector = FormDetector()
        self.associator = LabelAssociator()
        self.augmenter = PdfFormAugmenter()
        self.filler = FormFiller()
        self.refiner = MetadataRefiner()

    def get_page_mediabox(self, pdf_path: str, page_number: int):
        """Gets page mediabox and rotation from PDF in a single read."""
        import pypdf

        try:
            reader = pypdf.PdfReader(pdf_path)
            if 1 <= page_number <= len(reader.pages):
                page = reader.pages[page_number - 1]

                rotation = 0
                if hasattr(page, "rotation") and page.rotation is not None:
                    rotation = page.rotation
                elif "/Rotate" in page:
                    rotation = page["/Rotate"]
                rotation = int(rotation) % 360

                # Prefer CropBox if available, as it matches the visible area rendered by pdf2image
                mediabox = page.cropbox if "/CropBox" in page else page.mediabox
                return mediabox, rotation
        except Exception as e:
            print(f"Error reading PDF mediabox: {e}")
        return None, 0

    def get_existing_widgets(
        self,
        pdf_path: str,
        page_number: int,
        page_mediabox,
        rotation: int = 0,
        expected_width: float = None,
        expected_height: float = None,
    ) -> list[DetectedWidget]:
        """Extracts existing widgets from PDF annotations."""
        import pypdf

        widgets = []
        if not page_mediabox:
            return []

        page_left = float(page_mediabox.left)
        page_bottom = float(page_mediabox.bottom)
        page_width = float(page_mediabox.width)
        page_height = float(page_mediabox.height)

        # Determine current dimensions based on rotation
        if rotation in [90, 270]:
            current_width = page_height
            current_height = page_width
        else:
            current_width = page_width
            current_height = page_height

        scale_x = expected_width / current_width if expected_width and current_width > 0 else 1.0
        scale_y = (
            expected_height / current_height if expected_height and current_height > 0 else 1.0
        )

        try:
            reader = pypdf.PdfReader(pdf_path)
            if 1 <= page_number <= len(reader.pages):
                page = reader.pages[page_number - 1]
                if "/Annots" in page:
                    for annot in page["/Annots"]:
                        obj = annot.get_object()
                        if obj.get("/Subtype") == "/Widget":
                            field_name = obj.get("/T")
                            rect = obj.get("/Rect")  # [x_ll, y_ll, x_ur, y_ur]

                            # Convert PDF (Bottom-Left) to Image (Top-Left) coords
                            x_ll, y_ll, x_ur, y_ur = [float(c) for c in rect]

                            # Normalize to 0-based coordinates relative to mediabox
                            x_min = x_ll - page_left
                            y_min = y_ll - page_bottom
                            x_max = x_ur - page_left
                            y_max = y_ur - page_bottom

                            # Apply rotation transform
                            if rotation == 0:
                                x1 = x_min
                                y1 = page_height - y_max
                                x2 = x_max
                                y2 = page_height - y_min
                            elif rotation == 90:
                                x1 = y_min
                                y1 = x_min
                                x2 = y_max
                                y2 = x_max
                            elif rotation == 180:
                                x1 = page_width - x_max
                                y1 = y_min
                                x2 = page_width - x_min
                                y2 = y_max
                            elif rotation == 270:
                                x1 = page_height - y_max
                                y1 = page_width - x_max
                                x2 = page_height - y_min
                                y2 = page_width - x_min
                            else:
                                x1 = x_min
                                y1 = page_height - y_max
                                x2 = x_max
                                y2 = page_height - y_min

                            # Normalize bbox (x1 < x2, y1 < y2)
                            x1, x2 = min(x1, x2), max(x1, x2)
                            y1, y2 = min(y1, y2), max(y1, y2)

                            # Scale to expected dimensions
                            x1 *= scale_x
                            x2 *= scale_x
                            y1 *= scale_y
                            y2 *= scale_y

                            # Determine type
                            label = "text_input"
                            ft = obj.get("/FT")
                            if ft == "/Btn":
                                # Check flags for radio vs checkbox if needed, simplified here
                                label = "checkbox"
                            elif ft == "/Sig":
                                label = "signature"

                            # Extract value and filled status from PDF data
                            text_content = None
                            is_filled = False
                            if "/V" in obj:
                                raw_val = obj.get_object().get("/V")
                                if isinstance(raw_val, pypdf.generic.TextStringObject):
                                    text_content = str(raw_val)
                                    if text_content:
                                        is_filled = True
                                elif isinstance(raw_val, pypdf.generic.NameObject):
                                    if raw_val not in ("/Off", ""):
                                        is_filled = True
                                    if raw_val == "/Yes":
                                        text_content = "checked"
                                    elif raw_val != "/Off":
                                        # Clean up the name object string
                                        text_content = str(raw_val).lstrip("/")

                            widgets.append(
                                DetectedWidget(
                                    label=label,
                                    score=1.0,
                                    bbox=BoundingBox(x1, y1, x2, y2),
                                    is_filled=is_filled,
                                    text_content=text_content or "",
                                    is_existing=True,
                                    field_name=str(field_name) if field_name else None,
                                ),
                            )
        except Exception as e:
            print(f"Error extracting existing widgets: {e}")

        return widgets

    def process_page(
        self,
        page_data: PageData,
        source_pdf: str = "document.pdf",
        use_acroform: bool = True,
        use_widget_detection: bool = True,
    ) -> tuple[list[DetectedWidget], int, int]:
        print(f"Processing Page {page_data.page_number}...")
        total_input_tokens = 0
        total_output_tokens = 0

        # 0. Get Page Dimensions (mediabox and rotation read in a single PDF open)
        mediabox, rotation = self.get_page_mediabox(source_pdf, page_data.page_number)
        if mediabox:
            w, h = float(mediabox.width), float(mediabox.height)
            if rotation in [90, 270]:
                w, h = h, w
        else:
            # Fallback if PDF read fails, assume standard letter
            w, h = 612, 792

        # 1. Get Existing Widgets (Precedence)
        final_widgets = []
        if use_acroform:
            existing_widgets = self.get_existing_widgets(
                source_pdf,
                page_data.page_number,
                mediabox,
                rotation=rotation,
                expected_width=w,
                expected_height=h,
            )
            final_widgets.extend(existing_widgets)
            print(f"  Found {len(existing_widgets)} existing widgets in PDF.")

        # Helper to check for duplicates against accepted widgets
        def is_duplicate(widget, accepted_widgets):
            for accepted in accepted_widgets:
                # Check IoU
                if widget.bbox.iou(accepted.bbox) > 0.1:
                    return True
                # Check Containment (one inside the other)
                if widget.bbox.is_contained_in(accepted.bbox) or accepted.bbox.is_contained_in(
                    widget.bbox
                ):
                    return True
            return False

        # 3. Widget Detection
        if use_widget_detection and page_data.image_path and os.path.exists(page_data.image_path):
            gemini_widgets, in_tok, out_tok = self.detector.detect_gemini(
                page_data.image_path, w, h
            )
            total_input_tokens += in_tok
            total_output_tokens += out_tok
            added_count = 0
            for gw in gemini_widgets:
                if not is_duplicate(gw, final_widgets):
                    final_widgets.append(gw)
                    added_count += 1
            print(
                f"  Added {added_count} widgets from Widget Detection (discarded {len(gemini_widgets) - added_count} duplicates)."
            )
        elif use_widget_detection and page_data.image_path:
            print(f"  Image path specified but not found: {page_data.image_path}")

        print(f"  Total widgets for page: {len(final_widgets)}")

        # 3.5 Refine Metadata (VLM)
        in_tok, out_tok = self.refiner.refine(
            final_widgets, page_data.image_path, w, h, extract_context=False
        )
        total_input_tokens += in_tok
        total_output_tokens += out_tok

        # 4. Associate Labels
        for widget in final_widgets:
            widget.page_number = page_data.page_number

            # Extract surrounding text from fragments (geometric)
            geo_context = self._extract_geometric_context(widget, page_data.fragments)

            # Combine VLM context (if any) with geometric context
            if widget.surrounding_text:
                # Prioritize VLM context but keep geometric as fallback/augmentation
                widget.surrounding_text = f"{widget.surrounding_text} {geo_context}"
            else:
                widget.surrounding_text = geo_context

            # Only associate if a label hasn't already been assigned by the refiner
            if not widget.linked_text:
                linked_text = self.associator.associate(widget, page_data.fragments)
                widget.linked_text = linked_text

        # Sort widgets by position (Top-Left to Bottom-Right) to ensure reading order matching
        # This is critical for matching multiple identical placeholders in order
        final_widgets.sort(key=lambda w: (int(w.bbox.y1 / 10), int(w.bbox.x1)))

        return final_widgets, total_input_tokens, total_output_tokens

    def assign_field_names(self, widgets: list[DetectedWidget]):
        """Assigns deterministic field names to widgets before processing."""
        for w in widgets:
            if w.field_name:
                continue

            unique_suffix = f"{w.page_number}_{int(w.bbox.x1)}_{int(w.bbox.y1)}"
            if w.linked_text:
                safe_label = "".join(c for c in w.linked_text if c.isalnum() or c in ("_", "-"))
                field_name = f"{safe_label}_{unique_suffix}"
            else:
                field_name = f"field_{unique_suffix}"
            w.field_name = field_name

    def _extract_geometric_context(
        self, widget: DetectedWidget, fragments: list[PageFragment]
    ) -> str:
        """Finds text spatially close to the widget to use as context for anchoring."""
        w_center_y = widget.bbox.center[1]
        w_height = widget.bbox.height

        line_fragments = []
        for frag in fragments:
            if frag.fragment_type not in ["text", "title", "section_header", "list_item"]:
                continue

            f_center_y = frag.bbox.center[1]
            # Check if vertically aligned (roughly same line)
            if abs(f_center_y - w_center_y) < (max(w_height, frag.bbox.height) * 1.5):
                line_fragments.append(frag)

        # Sort by x coordinate to reconstruct the line
        line_fragments.sort(key=lambda f: f.bbox.x1)
        return " ".join([f.content for f in line_fragments])


# --- Ingestion Helpers ---


SECRETS = [
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_REGION",
]


@cls()
class FormFilling:
    @function(
        image=file_convertion_image,
        description="Fill a PDF form using AI.",
        timeout=30 * 60,  # 30 minutes
        cpu=2,
        memory=8,
        # output_encoder = "json"
        # The function is not using /tmp disk space, just reserve a small amount
        ephemeral_disk=2,
        secrets=SECRETS,
        min_containers=int(os.getenv("TENSORLAKE_MIN_CONTAINERS", "0")),
    )
    def run(self, result: ParseResult) -> ParsedDocumentRef:
        import pypdf

        request = result.request.form_filling

        source_pdf = result.request.file_bytes
        if isinstance(source_pdf, str):
            try:
                source_pdf = base64.b64decode(source_pdf)
            except Exception:
                pass

        if not source_pdf:
            raise ValueError("source_pdf is required")

        fill_prompt = request.fill_prompt
        ignore_source_values = request.ignore_source_values
        no_acroform = request.no_acroform
        no_widget_detection = request.no_widget_detection

        total_input_tokens = 0
        total_output_tokens = 0

        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = os.path.join(temp_dir, "input.pdf")
            output_pdf_path = os.path.join(temp_dir, "output.pdf")

            with open(input_path, "wb") as f:
                f.write(source_pdf)

            system = FormFillingSystem()

            from pdf2image import convert_from_path

            reader = pypdf.PdfReader(input_path)
            total_source_pdf_pages = len(reader.pages)
            pages_to_parse = (
                list(range(1, total_source_pdf_pages + 1))
                if not result.request.pages_to_parse
                else result.request.pages_to_parse
            )

            # Create a map for easy lookup
            pages_data_map = {
                i: PageData(page_number=i, fragments=[], image_path=None) for i in pages_to_parse
            }

            # Populate fragments from document_layout
            if result.document_layout:
                for page_layout in result.document_layout.pages:
                    if page_layout.page_number in pages_data_map:
                        page_data = pages_data_map[page_layout.page_number]
                        for element in page_layout.elements:
                            if element.ocr_text:
                                bbox_tuple = element.bbox
                                page_data.fragments.append(
                                    PageFragment(
                                        fragment_type=element.fragment_type.value,
                                        content=element.ocr_text,
                                        bbox=BoundingBox(
                                            x1=bbox_tuple[0],
                                            y1=bbox_tuple[1],
                                            x2=bbox_tuple[2],
                                            y2=bbox_tuple[3],
                                        ),
                                        reading_order=element.reading_order,
                                    )
                                )

            pages = list(pages_data_map.values())

            try:
                images = convert_from_path(input_path)
                for page in pages:
                    img_idx = page.page_number - 1
                    if 0 <= img_idx < len(images):
                        image_path = os.path.join(temp_dir, f"page_{page.page_number}.jpg")
                        images[img_idx].save(image_path, "JPEG")
                        page.image_path = image_path
            except Exception as e:
                print(f"Warning: PDF rendering failed: {e}")

            all_widgets = []
            pages_processed_count = 0
            for page in pages:
                pages_processed_count += 1
                widgets, in_tok, out_tok = system.process_page(
                    page,
                    source_pdf=input_path,
                    use_acroform=not no_acroform,
                    use_widget_detection=not no_widget_detection,
                )
                total_input_tokens += in_tok
                total_output_tokens += out_tok
                all_widgets.extend(widgets)

            if ignore_source_values:
                for w in all_widgets:
                    if w.is_existing:
                        w.text_content = None
                        w.is_filled = False

            system.assign_field_names(all_widgets)

            if fill_prompt:
                in_tok, out_tok = system.filler.fill(all_widgets, fill_prompt)
                total_input_tokens += in_tok
                total_output_tokens += out_tok

            system.augmenter.augment_pdf(input_path, output_pdf_path, all_widgets)

            out_pdf_b64 = None
            if os.path.exists(output_pdf_path):
                with open(output_pdf_path, "rb") as f:
                    out_pdf_bytes = f.read()
                out_pdf_b64 = base64.b64encode(out_pdf_bytes).decode("utf-8")

            widgets_data = [asdict(w) for w in all_widgets]
            metadata = {"detected_widgets": widgets_data}

            usage = Usage(
                pages_parsed=pages_processed_count,
                extraction_input_tokens_used=total_input_tokens,
                extraction_output_tokens_used=total_output_tokens,
            )

            result.usage = usage
            result.form_filling_result = FormFillingResult(
                filled_pdf_base64=out_pdf_b64,
                metadata=metadata,
            )

            return format_final_output(result)
