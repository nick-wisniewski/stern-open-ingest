# SPDX-License-Identifier: Apache-2.0
from typing import Dict, List, Union, Tuple, Optional

from tensorlake_docai.pipeline.api import (
    Page,
    PageFragment,
    PageFragmentType,
    Text,
    Table,
    Figure,
    ListItem,
    Signature,
    SectionHeader,
    ParseRequest,
    MergedTable,
)

from tensorlake_docai.models.layout_objects import PageLayout

from tensorlake.applications import (
    RequestError as RequestException,
)


def escape_markdown_content(content: str) -> str:
    """
    Escape markdown-sensitive characters in content to prevent interference with markdown formatting.

    Args:
        content: Raw text content that may contain markdown-sensitive characters

    Returns:
        Content with escaped markdown characters
    """
    if not content:
        return content

    # Split into lines to handle line-by-line escaping
    lines = content.split("\n")
    escaped_lines = []

    for line in lines:
        import re

        # Escape # at the beginning of lines (would be interpreted as headers)
        if line.lstrip().startswith("#"):
            # Replace # at start of line after any whitespace
            line = re.sub(r"^(\s*)(#+)", r"\1\\\2", line)

        # Escape # characters that are followed by space (typical header pattern)
        # Use separate patterns for start of line and after whitespace
        line = re.sub(r"\s(#+)(?=\s)", r" \\\1", line)  # # after whitespace + before space

        escaped_lines.append(line)

    return "\n".join(escaped_lines)


def escape_header_content(content: str) -> str:
    """
    Escape # characters in header content to prevent interference with markdown header markers.

    Args:
        content: Header text content

    Returns:
        Content with escaped # characters
    """
    if not content:
        return content

    # Escape all # characters in header content since they would interfere with header markers
    return content.replace("#", "\\#")


def page_fragment_to_markdown(page_fragment: PageFragment, request: ParseRequest) -> str:
    content = page_fragment.content

    if page_fragment.fragment_type == PageFragmentType.LIST_ITEM:
        escaped_content = escape_markdown_content(content.content)
        return f"* {escaped_content}\n"

    if page_fragment.fragment_type in [
        PageFragmentType.SECTION_HEADER,
        PageFragmentType.TITLE,
    ]:
        # Check if content already has markdown formatting
        content_text = content.content.strip()
        if content_text.startswith("#"):
            # Content already has markdown formatting, return as-is with proper spacing
            return f"\n{content_text}\n\n"

        # Use hierarchy level if available (SectionHeader), otherwise default to level 2
        if hasattr(content, "level"):
            markers = "#" * (content.level + 1)  # level 0->1, level 1->2, etc.
        else:
            markers = "##"  # Default for backward compatibility

        # Escape # characters in header content to prevent conflicts with header markers
        escaped_header_content = escape_header_content(content.content)
        return f"\n{markers} {escaped_header_content}\n\n"

    if page_fragment.fragment_type == PageFragmentType.TEXT:
        escaped_content = escape_markdown_content(content.content)
        return f"{escaped_content}\n\n"

    if page_fragment.fragment_type in [
        PageFragmentType.FORMULA,
        PageFragmentType.FORMULA_CAPTION,
        PageFragmentType.TABLE_CAPTION,
        PageFragmentType.FIGURE_CAPTION,
    ]:
        escaped_content = escape_markdown_content(content.content)
        return f"{escaped_content}\n\n"

    if page_fragment.fragment_type in [PageFragmentType.FIGURE, PageFragmentType.CHART]:
        escaped_figure_content = escape_markdown_content(content.content)
        return f"### Figure \n{escaped_figure_content}\n\n"

    if page_fragment.fragment_type == PageFragmentType.TABLE:
        table_summary = ""
        if content.summary:
            escaped_summary = escape_markdown_content(content.summary)
            table_summary = f"Table Summary \n{escaped_summary}\n\n"
        table_content = (
            f"{content.html}" if request.table_output_mode == "html" else f"{content.markdown}"
        )
        return f"\n{table_content}\n{table_summary}\n"

    # Default case for unexpected fragment types
    escaped_content = escape_markdown_content(content.content)
    return f"\n{escaped_content}\n"


def page_to_markdown(page: Page, request: ParseRequest) -> str:
    text = ""
    fragments = page.page_fragments
    for fragment in fragments:
        text += page_fragment_to_markdown(fragment, request)
    return text


def document_to_markdown(pages: List[Page], request: ParseRequest) -> str:
    text = ""
    for page in pages:
        text += page_to_markdown(page, request)
        text += "\n\n"

    return text


def _downsample_bbox_coordinates(bbox, scale_factor):
    if bbox is None:
        return bbox

    scaled_bbox = {}
    for k in bbox.keys():
        int_cord = int(bbox[k] // scale_factor)
        scaled_bbox[k] = int_cord
    return scaled_bbox


def _bbox_to_dict(bbox: Tuple[float, float, float, float]) -> Dict[str, float]:
    return {"x1": bbox[0], "y1": bbox[1], "x2": bbox[2], "y2": bbox[3]}


def document_layout_to_document(
    page_layouts: List[PageLayout],
    scale_factor: float,
    ignore_sections: Optional[set] = None,
    merged_tables: Optional[List[MergedTable]] = None,
    chunking_strategy: str = "none",
) -> List[Page]:
    from markdownify import markdownify

    parsed_pages: List[Page] = []
    for page_layout in page_layouts:
        parsed_page_fragments = []
        for page_element in page_layout.elements:
            content = page_element.ocr_text

            if content is None:
                print(f"Skipping empty ocr_text element of class {page_element.fragment_type}")
                continue

            # Skip elements that are in the ignore_sections set
            if ignore_sections and page_element.fragment_type in ignore_sections:
                print(f"Skipping ignored element of class {page_element.fragment_type}")
                continue

            parsed_content: Union[Text, Table, Figure, SectionHeader]
            if page_element.fragment_type in [
                PageFragmentType.SECTION_HEADER,
                PageFragmentType.TITLE,
            ]:
                # Create SectionHeader with hierarchy level if available
                if (
                    hasattr(page_element, "hierarchy_level")
                    and page_element.hierarchy_level is not None
                ):
                    parsed_content = SectionHeader(
                        content=content, level=page_element.hierarchy_level
                    )
                else:
                    # Fallback to regular Text for backward compatibility
                    parsed_content = Text(content=content)
            elif page_element.fragment_type in [
                PageFragmentType.TEXT,
                PageFragmentType.FORMULA,
                PageFragmentType.FORMULA_CAPTION,
                PageFragmentType.TABLE_CAPTION,
                PageFragmentType.FIGURE_CAPTION,
                PageFragmentType.PAGE_FOOTER,
                PageFragmentType.PAGE_HEADER,
                PageFragmentType.PAGE_NUMBER,
                PageFragmentType.COMMENTS,
                PageFragmentType.TRACKED_CHANGES,
                PageFragmentType.BARCODE,
            ]:
                parsed_content = Text(
                    content=content,
                    html=page_element.html,
                )
            elif page_element.fragment_type in [
                PageFragmentType.TABLE,
                PageFragmentType.KEY_VALUE_REGION,
                PageFragmentType.FORM,
            ]:
                parsed_content = Table(
                    content=content,
                    html=page_element.html,
                    markdown=page_element.markdown,
                    summary=page_element.llm_summary,
                    table_checked=(
                        page_element.table_checked
                        if hasattr(page_element, "table_checked")
                        else False
                    ),
                )
            elif page_element.fragment_type in [
                PageFragmentType.LIST_ITEM,
                PageFragmentType.DOCUMENT_INDEX,
            ]:
                parsed_content = ListItem(
                    content=content,
                )
            elif page_element.fragment_type in [PageFragmentType.FIGURE, PageFragmentType.CHART]:
                parsed_content = Figure(
                    content=content,
                )
            elif page_element.fragment_type == PageFragmentType.SIGNATURE:
                parsed_content = Signature(
                    content=content,
                )

            else:
                raise ValueError(f"Unknown fragment type: {page_element.fragment_type}")

            parsed_page_fragments.append(
                PageFragment(
                    fragment_type=page_element.fragment_type,
                    content=parsed_content,
                    bbox=_downsample_bbox_coordinates(
                        _bbox_to_dict(page_element.bbox), scale_factor
                    ),
                    reading_order=page_element.reading_order,
                    ref_id=page_element.ref_id if hasattr(page_element, "ref_id") else None,
                )
            )

        parsed_pages.append(
            Page(
                page_number=page_layout.page_number,
                page_fragments=parsed_page_fragments,
                dimensions=page_layout.shape,
                page_dimensions=page_layout.page_dimensions,
            )
        )
    parsed_pages.sort(key=lambda x: x.page_number)

    if merged_tables:
        if chunking_strategy == "page":
            raise RequestException("Table merging is not supported with 'page' chunking strategy.")

        ref_ids_to_remove = set()
        page_to_merged_tables = {}

        for mt in merged_tables:
            if mt.merge_actions and mt.merge_actions.ref_ids:
                ref_ids_to_remove.update(mt.merge_actions.ref_ids)

            # Determine which pages to add the merged table to
            target_pages = [mt.start_page]

            for p in target_pages:
                if p not in page_to_merged_tables:
                    page_to_merged_tables[p] = []
                page_to_merged_tables[p].append(mt)

        for page in parsed_pages:
            # Remove fragments
            if page.page_fragments:
                new_fragments = [
                    f
                    for f in page.page_fragments
                    if not (f.ref_id and f.ref_id in ref_ids_to_remove)
                ]
                page.page_fragments = new_fragments

            # Add merged tables
            if page.page_number in page_to_merged_tables:
                if page.page_fragments is None:
                    page.page_fragments = []

                for mt in page_to_merged_tables[page.page_number]:
                    # Convert HTML to markdown
                    mt_markdown = markdownify(mt.merged_table_html)

                    table_content = Table(
                        content=mt.merged_table_html,
                        html=mt.merged_table_html,
                        markdown=mt_markdown,
                        summary=mt.summary,
                    )

                    frag = PageFragment(
                        fragment_type=PageFragmentType.TABLE,
                        content=table_content,
                        bbox=None,
                        reading_order=1000000,  # Append at end
                        ref_id=mt.merged_table_id,
                    )
                    page.page_fragments.append(frag)

    return parsed_pages
