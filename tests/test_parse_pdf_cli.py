# SPDX-License-Identifier: Apache-2.0
import urllib.error

import pytest

from examples import parse_pdf


def test_parser_accepts_paddle_ocr_vl():
    args = parse_pdf.build_parser().parse_args(
        [
            "--file",
            "sample.pdf",
            "--ocr-model",
            "paddle-ocr-vl",
            "--pages",
            "1",
            "--local",
        ]
    )

    assert args.ocr_model == "paddle-ocr-vl"
    assert args.pages == [1]


def test_build_request_sets_paddle_ocr_model(tmp_path):
    sample = tmp_path / "sample.pdf"
    sample.write_bytes(b"%PDF-1.4\n")
    args = parse_pdf.build_parser().parse_args(
        [
            "--file",
            str(sample),
            "--ocr-model",
            "paddle-ocr-vl",
            "--pages",
            "1",
        ]
    )

    request = parse_pdf.build_request(args)

    assert request.ocr_model == "paddle-ocr-vl"
    assert request.pages_to_parse == [1]


def test_paddle_preflight_reports_unreachable_server(monkeypatch):
    from tensorlake_docai.ocr import paddle_ocr_vl

    monkeypatch.setattr(paddle_ocr_vl, "_cuda_is_available", lambda: True)
    monkeypatch.setenv("PADDLE_OCR_VL_SERVER_URL", "http://127.0.0.1:8118/v1")

    def fail_urlopen(*args, **kwargs):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(parse_pdf.urllib.request, "urlopen", fail_urlopen)

    with pytest.raises(RuntimeError, match="recognition server is not reachable"):
        parse_pdf.check_paddle_preflight("paddle-ocr-vl")


def test_paddle_preflight_skips_other_models(monkeypatch):
    def fail_urlopen(*args, **kwargs):
        raise AssertionError("preflight should not contact server for dots-ocr")

    monkeypatch.setattr(parse_pdf.urllib.request, "urlopen", fail_urlopen)

    parse_pdf.check_paddle_preflight("dots-ocr")
