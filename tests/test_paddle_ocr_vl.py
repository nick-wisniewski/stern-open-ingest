# SPDX-License-Identifier: Apache-2.0
from tensorlake_docai.ocr.paddle_ocr_vl import (
    map_paddle_label_to_fragment_type,
    paddle_result_to_page_layout,
)
from tensorlake_docai.pipeline.api import PageFragmentType


def test_paddle_label_mapping_covers_common_blocks():
    assert map_paddle_label_to_fragment_type("doc_title") == PageFragmentType.TITLE
    assert map_paddle_label_to_fragment_type("paragraph_title") == PageFragmentType.SECTION_HEADER
    assert map_paddle_label_to_fragment_type("table") == PageFragmentType.TABLE
    assert map_paddle_label_to_fragment_type("image") == PageFragmentType.FIGURE
    assert map_paddle_label_to_fragment_type("unknown") == PageFragmentType.TEXT


def test_paddle_result_to_page_layout_scales_and_orders_blocks():
    paddle_result = {
        "res": {
            "parsing_res_list": [
                {
                    "block_bbox": [20, 20, 100, 80],
                    "block_label": "table",
                    "block_content": "<table><tr><td>A</td></tr></table>",
                    "block_id": 2,
                    "block_order": 2,
                    "score": 0.8,
                },
                {
                    "block_bbox": [10, 10, 120, 18],
                    "block_label": "doc_title",
                    "block_content": "Policy Declaration",
                    "block_id": 1,
                    "block_order": 1,
                    "score": 0.9,
                },
            ]
        }
    }

    layout = paddle_result_to_page_layout(
        paddle_result,
        page_number=3,
        image_size=(200, 100),
        pdf_size=(100, 50),
    )

    assert layout.page_number == 3
    assert layout.shape == (100, 50)
    assert [element.reading_order for element in layout.elements] == [1, 2]
    assert layout.elements[0].fragment_type == PageFragmentType.TITLE
    assert layout.elements[0].bbox == (5.0, 5.0, 60.0, 9.0)
    assert layout.elements[0].ref_id == "3.1"
    assert layout.elements[1].fragment_type == PageFragmentType.TABLE
    assert layout.elements[1].html == "<table><tr><td>A</td></tr></table>"
    assert "A" in layout.elements[1].markdown


def test_paddle_result_to_page_layout_falls_back_to_layout_boxes():
    paddle_result = {
        "layout_det_res": {
            "boxes": [
                {
                    "label": "text",
                    "coordinate": [0, 0, 50, 20],
                    "score": 0.7,
                }
            ]
        }
    }

    layout = paddle_result_to_page_layout(
        paddle_result,
        page_number=1,
        image_size=(50, 20),
        pdf_size=(50, 20),
    )

    assert len(layout.elements) == 1
    assert layout.elements[0].fragment_type == PageFragmentType.TEXT
    assert layout.elements[0].ocr_text == ""
