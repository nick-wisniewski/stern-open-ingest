# SPDX-License-Identifier: Apache-2.0
"""Registry-table invariants — guards against silently dropping a provider
or accidentally re-introducing the dropped `model06` (Marker)."""

from tensorlake_docai.ocr import OCR_BACKENDS, DEFAULT_OCR_MODEL


def test_supported_models():
    assert set(OCR_BACKENDS) == {
        "dots-ocr",
    }


def test_cloud_backends_are_dropped():
    # The Azure / Textract / Gemini cloud OCR backends were removed from this
    # fork; only the self-hosted dots-ocr engine remains.
    for dropped in ("azure-di", "textract", "gemini", "model06"):
        assert dropped not in OCR_BACKENDS


def test_backend_class_paths_stable():
    # Stringified class paths — the pipeline imports these lazily so that
    # GPU-only modules don't load in non-GPU workers.
    assert OCR_BACKENDS["dots-ocr"] == "tensorlake_docai.ocr.dots_ocr.DotsOCRTask"


def test_default_model_is_in_registry():
    assert DEFAULT_OCR_MODEL in OCR_BACKENDS
