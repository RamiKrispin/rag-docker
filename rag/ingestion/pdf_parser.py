from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from docling.document_converter import DocumentConverter

__all__ = ["ParsedElement", "parse_pdf"]


@dataclass
class ParsedElement:
    content: str
    type: str  # "text", "table", "heading"
    page: int
    section_title: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def parse_pdf(path: str | Path) -> list[ParsedElement]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    converter = DocumentConverter()
    result = converter.convert(str(path))
    doc = result.document

    elements: list[ParsedElement] = []
    current_section = ""

    for item in doc.iterate_items():
        element, _level, *_ = item
        page_num = _get_page_number(element, doc)

        if hasattr(element, "label"):
            label = element.label.value if hasattr(
                element.label, "value"
            ) else str(element.label)
        else:
            label = "text"

        # Skip picture/image elements
        if label in ("picture", "figure"):
            continue

        if label in ("section_header", "title"):
            current_section = _get_text(element, doc)
            elements.append(ParsedElement(
                content=current_section,
                type="heading",
                page=page_num,
                section_title=current_section,
            ))
        elif label == "table":
            table_text = _export_table(element, doc)
            elements.append(ParsedElement(
                content=table_text,
                type="table",
                page=page_num,
                section_title=current_section,
            ))
        else:
            text = _get_text(element, doc)
            if text.strip():
                elements.append(ParsedElement(
                    content=text,
                    type="text",
                    page=page_num,
                    section_title=current_section,
                ))

    return elements


def _get_text(element, doc=None) -> str:
    if hasattr(element, "text"):
        return element.text
    if hasattr(element, "export_to_markdown"):
        try:
            return element.export_to_markdown()
        except TypeError:
            if doc is not None:
                return element.export_to_markdown(doc)
    return str(element)


def _export_table(element, doc=None) -> str:
    if hasattr(element, "export_to_markdown"):
        try:
            return element.export_to_markdown()
        except TypeError:
            if doc is not None:
                return element.export_to_markdown(doc)
    if hasattr(element, "text"):
        return element.text
    return str(element)


def _get_page_number(element, doc) -> int:
    if hasattr(element, "prov") and element.prov:
        for prov in element.prov:
            if hasattr(prov, "page_no"):
                return prov.page_no
    return 0
