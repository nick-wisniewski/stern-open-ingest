# SPDX-License-Identifier: Apache-2.0
"""
Entrypoint that registers every workflow function for the Open Ingest pipeline.

Importing this module wires up all the `@function()`/`@cls()` tasks so the
`--local` runner can dispatch them in-process. This file must sit ONE LEVEL
ABOVE the `tensorlake_docai/` package (i.e. at `src/workflow.py`, not inside
`src/tensorlake_docai/`) so absolute imports like
`from tensorlake_docai.vlm.cloud import ...` resolve consistently.

Usage:
    pip install -e .
    python examples/parse_pdf.py --file my.pdf --local
"""

# Application entry — file conversion + OCR routing.
from tensorlake_docai.pipeline.file_converter import normalize_file_type_and_upload  # noqa: F401

# `dots-ocr` GPU path — requires a CUDA-equipped worker (vLLM + CUDA, multi-GB
# image). Disabled by default so a non-GPU runner does not have to build the
# heavy `ocr-gpu-cuda` image. Re-enable both imports once you have a GPU host.
# from tensorlake_docai.ocr.dots_ocr import DotsOCRTask  # noqa: F401
# from tensorlake_docai.ocr.figure_ocr import OvisFigureOCRTask  # noqa: F401

# Post-OCR enrichment.
from tensorlake_docai.tables.table_merging import TableMerging  # noqa: F401
from tensorlake_docai.vlm.cloud import VLMExtractionTask  # noqa: F401

# Output formatting (terminal node).
from tensorlake_docai.pipeline.output_formatter import format_final_output  # noqa: F401
