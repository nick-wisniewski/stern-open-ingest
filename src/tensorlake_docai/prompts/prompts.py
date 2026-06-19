# SPDX-License-Identifier: Apache-2.0
# pylint: disable=line-too-long
# flake8: noqa: E501
from tensorlake_docai.pipeline.api import PageFragmentType

TABLE_EXTRACTOR_SYSTEM_PROMPT = (
    """You are a helpful AI assistant specialized in extracting table data from images."""
)

TABLE_EXTRACTOR_USER_PROMPT = """[INST] Extract information from the table image as structured data. 
* Format the output as markdown. 
* Strictly return the extracted output only, and nothing else. Don't add any other text in the beggining of the output.
* If there are radio boxes or checkboxes, only include the text around the checkbox thats checked or marked. If none of the checkboxes are checked, return an empty string.
* INCLUDE ALL the information on the table.
* Don't add backticks or any other formating information like ```markdown or ```json in the output
[/INST]
"""

FIGURE_EXTRACTOR_SYSTEM_PROMPT = (
    """You are a helpful AI assistant specialized in extracting figure data from images."""
)

FIGURE_EXTRACTOR_USER_PROMPT = """Extract information from the figure:

* If it is a table: Convert to structured markdown with title and caption
* If it is a graph or chart: Identify type and describe key data trends
* If it contains text: Extract all printed or handwritten content
* If it's a mathematical formula: Transcribe using LaTeX notation
* If it has checkboxes: Only include text for selected options (empty string if none)

Return only extracted information in markdown format without additional commentary.
"""


KEY_VALUE_EXTRACTOR_USER_PROMPT = """[INST] Extract information from the document fragment:

* If it is a table, extract the table data into a structured markdown format, along with table title and caption. 
* If it is a graph or chart, describe it in detail.
* If it contains printed or handwritten text, extract all the text.
* If there are radio boxes or checkboxes, only include the text around the checkbox thats checked or marked. If none of the checkboxes are checked, return an empty string.
* If it contains labeled fields or key-value pairs, extract them into structured markdown with question and answer pairs.
Don't add ```text or ```markdown in the output.
[/INST]
"""

DOCUMENT_INDEX_SYSTEM_PROMPT = (
    """You are a helpful AI assistant specialized in extracting document index from images."""
)

DOCUMENT_INDEX_USER_PROMPT = """[INST] This is a cropped image from a document. This is a document index.Extract information from the document index:
* Extract the document index from the image.
* Return the document index in a structured markdown format. Don't include any additional markup like ```markdown or ```json.
[/INST]
"""

OCR_SYSTEM_PROMPT = """You are an OCR modelspecialized in extracting text from images."""
OCR_PROMPT = """
(<image>./</image>)
[INST]
Extract the text in the image. If this is written, printed or text, only extract the text. If it's a picture with some text on the scene, extract only the text. 
Don't describe the image. 
Don't add any other text or information at the beginning of the text.
Don't add ```text or ```markdown in the output.
ABSOLUTELY DON'T REPEAT THE TEXT OR HALLUCINATE ANYTHING. 
IF THE IMAGE IS EMPTY RETURN |notext| 
[/INST]
"""

TABLE_MERGING_PROMPTS = {
    "merged_summary": {
        "system": (
            "You are a helpful assistant that summarizes merged tables. "
            "Given a merged HTML table, provide a concise summary of its contents."
        ),
        "user": "Given the following merged HTML table:\n{merged_table}\n\n"
        "Provide a concise summary of the table's contents, highlighting key information and any notable patterns. "  # noqa E501
        'Return the output as JSON in the format: {{\n  "summary": "concise summary here"\n}}',  # noqa E501
    },
    "fast_merge": {
        "system": (
            "You are a helpful assistant that analyzes HTML tables to determine if they should be merged. "
            "The output is valid JSON."
        ),
        "user": "Given the following HTML table from the end of the first page:\n{table_end}\n\n"
        "Context between tables (e.g. page footers/headers):\n{context_between}\n\n"
        "And the following HTML table from the start of the second page:\n{table_start}\n\n"
        "Step 1: Determine if the second table is a continuation of the first table.\n"
        "Step 2: If 'YES', provide an explanation as to why and the number of rows to skip from the start of the second table (0 if none). This refers to header rows that could be removed to improve merging.\n"
        "If 'NO', do not merge.\n\n"
        "Answer in the following JSON format:\n"
        "{{\n"
        "  'continuation': 'YES' or 'NO',\n"
        "  'explanation': 'brief explanation',\n"
        "  'skip_rows': 'number of rows to skip.'\n"
        "}}",
    },
    "fast_same_page_merge": {
        "system": (
            "You are a helpful assistant that analyzes HTML tables on the same page to determine if they should be merged. "
            "The output is valid JSON."
        ),
        "user": "Table 1:\n{table1}\n\n"
        "Context between tables (e.g. page footers/headers):\n{context_between}\n\n"
        "Table 2:\n{table2}\n\n"
        "Step 1: Determine if Table 2 is a continuation of Table 1 (e.g. split by layout).\n"
        "Step 2: If 'YES', determine how many header rows at the beginning of Table 2 repeat headers from Table 1 and should be skipped during merging.\n"
        "Step 3: If 'YES' and the column structure of Table 2 appears misaligned with Table 1, provide a corrected HTML for the *entire* Table 2 to ensure it aligns properly. For the alignment, pay attention to the type of content of the columns both syntactic and semantically, so they two tables are best aligned. If no correction is needed, this field can be null.\n"
        "If 'NO', do not merge.\n\n"
        "Answer in the following JSON format:\n"
        "{{\n"
        "  'should_merge': 'YES' or 'NO',\n"
        "  'explanation': 'brief explanation',\n"
        "  'skip_rows': <number of rows to skip from the start of Table 2 (0 if none)>,\n"
        "  'corrected_html': <corrected HTML for Table 2 if needed, otherwise null>\n"
        "}}",
    },
    "align_tables": {
        "system": "You are an expert in semantically aligning HTML table columns based on their visual representation for merging tables.",
        "user": "The following two tables are from consecutive pages of a document and are part of the same semantic table. "
        "However, their column structures in HTML do not align semantically.\n"
        "Table 1 (End of previous page):\n"
        " {table1_html} \n\n"
        "Table 2 (Start of current page):\n"
        " {table2_html} \n\n"
        "You are provided with images of both tables.\n"
        "Your task is to generate a CORRECTED HTML for Table 2 so that its columns align perfectly semantically with Table 1.\n"
        "- Ensure Table 2 has the same number of columns as Table 1.\n"
        "- Adjust column spans or add empty cells or remove cells if necessary to match the semantic alignment.\n"
        "- The content of Table 2 must be preserved, just restructured cells to align with Table 1.\n"
        "- Do not repeat content not present in the Table 2 image.\n"
        "- Return the corrected HTML and an explanation as JSON in the format: {{ \"corrected_html': '<html>...</html>\", \"explanation\": '<explanation>'}}\n",
    },
}

TABLE_CORRECTION_PROMPTS = {
    "system": (
        "You are an expert in HTML table correction. "
        "You will receive an image of a table, an extracted HTML version, and an error analysis report. "
        "Your task is to fix the HTML table to match the image and resolve the reported errors."
    ),
    "user": "You are an expert in HTML table correction. "
    "You will receive an image of a table and an HTML version. "
    "Your task is to fix the HTML table to match the image and resolve the reported errors and any other error. "
    "Analyze the structure of the table in the image, look at table lines and gaps, and use it as reference to correct the HTML so the rows and columns are reflected accurately. "
    "Use the table headers, cell contents, and overall layout as clues to fix merged or split cells, misaligned data, and any other issues. "
    "The HTML does not need to contain any style or formatting, focus solely on the structure and content accuracy. "
    "In the cell content, there is no need to include characters when added to fill empty spaces, e.g. with dots or hyphens.\n\n"
    "HTML Table:\n{html_input}\n\n"
    "Error Analysis Report:\n{error_report}\n\n"
    'The output is in JSON with a "corrected_html" field and an "explanation" field.',
}

KEY_VALUE_PROMPTS = {
    "detection": {
        "system": "You are an expert at detecting key-value regions in document images.",
        "user": "Analyze this image and determine if it contains labeled fields or key-value pairs. "
        "Return a JSON object with a single boolean field 'is_key_value_region'.",
    },
    "extraction": {
        "system": "You are an expert at extracting key-value data from document regions.",
        "user": "Extract the information from this region. Represent key-value pairs clearly. "
        "For checkboxes/radio buttons, indicate their state (e.g., [x] for checked, [ ] for unchecked). "
        "Consider field labels and surrounding context when processing the image. "
        "Preserve the structure and order as much as possible.",
    },
}


QWEN_TABLE_TO_MARKDOWN_SYSTEM_PROMPT = "You are an AI assistant specialized in recognizing and extracting text from images. Your mission is to analyze the image and generate result in markdown and only use the text in the image."
QWEN_TABLE_TO_MARKDOWN_PROMPT = "Convert the document fragment to markdown. Do not include the text not on the image, do not include other outputs."

# HTML versions of table prompts
QWEN_TABLE_TO_HTML_SYSTEM_PROMPT = "You are an AI assistant specialized in recognizing and extracting text from images. Your mission is to analyze the image and generate result in HTML and only use the text in the image."
QWEN_TABLE_TO_HTML_PROMPT = "Convert the document fragment to HTML. Do not include the text not on the image, do not include other outputs."


def get_prompt_messages(cls: PageFragmentType) -> list[str]:
    if cls in [PageFragmentType.TABLE]:
        return [TABLE_EXTRACTOR_SYSTEM_PROMPT, TABLE_EXTRACTOR_USER_PROMPT]
    elif cls in [PageFragmentType.FIGURE, PageFragmentType.FORMULA]:
        return [FIGURE_EXTRACTOR_SYSTEM_PROMPT, FIGURE_EXTRACTOR_USER_PROMPT]
    elif cls in [PageFragmentType.FORM, PageFragmentType.KEY_VALUE_REGION]:
        return [TABLE_EXTRACTOR_SYSTEM_PROMPT, KEY_VALUE_EXTRACTOR_USER_PROMPT]
    elif cls in [PageFragmentType.DOCUMENT_INDEX]:
        return [DOCUMENT_INDEX_SYSTEM_PROMPT, DOCUMENT_INDEX_USER_PROMPT]
    elif cls in [
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
    ]:
        return [OCR_SYSTEM_PROMPT, OCR_PROMPT]
    else:
        raise ValueError(f"Invalid class name: {cls}")


def _get_prompt_messages_from_dict(task: str, prompt_dict: dict, task_type: str) -> list[str]:
    if task in prompt_dict:
        prompt_data = prompt_dict[task]
        return [prompt_data["system"], prompt_data["user"]]
    else:
        raise ValueError(f"Invalid {task_type} task: {task}")


def get_table_merging_prompt_messages(task: str) -> list[str]:
    return _get_prompt_messages_from_dict(task, TABLE_MERGING_PROMPTS, "table merging")


def get_table_correction_prompt_messages(additional_prompt: str | None = None) -> list[str]:
    return TABLE_CORRECTION_PROMPTS["system"], TABLE_CORRECTION_PROMPTS["user"]


def get_key_value_prompt_messages(task: str) -> list[str]:
    return _get_prompt_messages_from_dict(task, KEY_VALUE_PROMPTS, "key-value extraction")
