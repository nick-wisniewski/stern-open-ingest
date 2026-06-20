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

import os

# Application entry — file conversion + OCR routing.
from tensorlake_docai.pipeline.file_converter import normalize_file_type_and_upload  # noqa: F401

# GPU OCR paths require CUDA-equipped workers and multi-GB images. Keep them
# disabled by default for CPU-only local runners, but make GPU registration
# repeatable for smoke tests and worker launches.
if os.getenv("ENABLE_GPU_OCR_TASKS") == "1":
    from tensorlake_docai.ocr.dots_ocr import DotsOCRTask  # noqa: F401
    from tensorlake_docai.ocr.figure_ocr import OvisFigureOCRTask  # noqa: F401
    from tensorlake_docai.ocr.paddle_ocr_vl import PaddleOCRVLTask  # noqa: F401

# Post-OCR enrichment.
from tensorlake_docai.tables.table_merging import TableMerging  # noqa: F401
from tensorlake_docai.vlm.cloud import VLMExtractionTask  # noqa: F401

# Output formatting (terminal node).
from tensorlake_docai.pipeline.output_formatter import format_final_output  # noqa: F401
