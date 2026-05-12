from dataclasses import dataclass, field
from typing import Any

from rag.ingestion.pdf_parser import ParsedElement

__all__ = ["Chunk", "chunk_elements"]


@dataclass
class Chunk:
    content: str
    type: str  # "text", "table", "heading"
    page: int
    section_title: str = ""
    chunk_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


def chunk_elements(
    elements: list[ParsedElement],
    *,
    method: str = "recursive",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    keep_tables_intact: bool = True,
) -> list[Chunk]:
    allowed = {"recursive", "semantic", "by_title"}
    if method not in allowed:
        raise ValueError(
            f"chunking method must be one of {allowed}, got '{method}'"
        )

    if method == "recursive":
        return _chunk_recursive(
            elements, chunk_size, chunk_overlap, keep_tables_intact
        )
    elif method == "by_title":
        return _chunk_by_title(
            elements, chunk_size, chunk_overlap, keep_tables_intact
        )
    elif method == "semantic":
        return _chunk_semantic(
            elements, chunk_size, chunk_overlap, keep_tables_intact
        )
    else:
        raise ValueError(
            f"chunking method must be one of {allowed}, got '{method}'"
        )


def _chunk_recursive(
    elements: list[ParsedElement],
    chunk_size: int,
    chunk_overlap: int,
    keep_tables_intact: bool,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    chunk_idx = 0

    for elem in elements:
        if elem.type == "table" and keep_tables_intact:
            chunks.append(Chunk(
                content=elem.content,
                type="table",
                page=elem.page,
                section_title=elem.section_title,
                chunk_index=chunk_idx,
            ))
            chunk_idx += 1
            continue

        if len(elem.content) <= chunk_size:
            chunks.append(Chunk(
                content=elem.content,
                type=elem.type,
                page=elem.page,
                section_title=elem.section_title,
                chunk_index=chunk_idx,
            ))
            chunk_idx += 1
        else:
            splits = _split_text(
                elem.content, chunk_size, chunk_overlap
            )
            for split in splits:
                chunks.append(Chunk(
                    content=split,
                    type=elem.type,
                    page=elem.page,
                    section_title=elem.section_title,
                    chunk_index=chunk_idx,
                ))
                chunk_idx += 1

    return chunks


def _chunk_by_title(
    elements: list[ParsedElement],
    chunk_size: int,
    chunk_overlap: int,
    keep_tables_intact: bool,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    chunk_idx = 0
    current_group: list[ParsedElement] = []
    current_section = ""

    for elem in elements:
        if elem.type == "heading":
            if current_group:
                chunks.extend(_flush_group(
                    current_group, current_section,
                    chunk_size, chunk_overlap,
                    keep_tables_intact, chunk_idx
                ))
                chunk_idx = len(chunks)
            current_section = elem.content
            current_group = []
        else:
            current_group.append(elem)

    if current_group:
        chunks.extend(_flush_group(
            current_group, current_section,
            chunk_size, chunk_overlap,
            keep_tables_intact, chunk_idx
        ))

    return chunks


def _chunk_semantic(
    elements: list[ParsedElement],
    chunk_size: int,
    chunk_overlap: int,
    keep_tables_intact: bool,
) -> list[Chunk]:
    # Semantic chunking groups consecutive elements by similarity.
    # As a baseline, we group adjacent text elements until chunk_size
    # is reached, breaking at paragraph boundaries.
    chunks: list[Chunk] = []
    chunk_idx = 0
    buffer = ""
    buffer_page = 0
    buffer_section = ""

    for elem in elements:
        if elem.type == "table" and keep_tables_intact:
            if buffer.strip():
                chunks.append(Chunk(
                    content=buffer.strip(),
                    type="text",
                    page=buffer_page,
                    section_title=buffer_section,
                    chunk_index=chunk_idx,
                ))
                chunk_idx += 1
                buffer = ""

            chunks.append(Chunk(
                content=elem.content,
                type="table",
                page=elem.page,
                section_title=elem.section_title,
                chunk_index=chunk_idx,
            ))
            chunk_idx += 1
            continue

        if not buffer:
            buffer_page = elem.page
            buffer_section = elem.section_title

        candidate = buffer + "\n\n" + elem.content if buffer else elem.content

        if len(candidate) <= chunk_size:
            buffer = candidate
        else:
            if buffer.strip():
                chunks.append(Chunk(
                    content=buffer.strip(),
                    type="text",
                    page=buffer_page,
                    section_title=buffer_section,
                    chunk_index=chunk_idx,
                ))
                chunk_idx += 1
                overlap_text = buffer.strip()[-chunk_overlap:]
                buffer = overlap_text + "\n\n" + elem.content
            else:
                buffer = elem.content
            buffer_page = elem.page
            buffer_section = elem.section_title

    if buffer.strip():
        chunks.append(Chunk(
            content=buffer.strip(),
            type="text",
            page=buffer_page,
            section_title=buffer_section,
            chunk_index=chunk_idx,
        ))

    return chunks


def _flush_group(
    group: list[ParsedElement],
    section: str,
    chunk_size: int,
    chunk_overlap: int,
    keep_tables_intact: bool,
    start_idx: int,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    idx = start_idx

    combined = ""
    page = group[0].page if group else 0

    for elem in group:
        if elem.type == "table" and keep_tables_intact:
            if combined.strip():
                for split in _split_text(
                    combined.strip(), chunk_size, chunk_overlap
                ):
                    chunks.append(Chunk(
                        content=split,
                        type="text",
                        page=page,
                        section_title=section,
                        chunk_index=idx,
                    ))
                    idx += 1
                combined = ""
            chunks.append(Chunk(
                content=elem.content,
                type="table",
                page=elem.page,
                section_title=section,
                chunk_index=idx,
            ))
            idx += 1
        else:
            combined += "\n\n" + elem.content if combined else elem.content

    if combined.strip():
        for split in _split_text(
            combined.strip(), chunk_size, chunk_overlap
        ):
            chunks.append(Chunk(
                content=split,
                type="text",
                page=page,
                section_title=section,
                chunk_index=idx,
            ))
            idx += 1

    return chunks


def _split_text(
    text: str, chunk_size: int, chunk_overlap: int
) -> list[str]:
    if len(text) <= chunk_size:
        return [text]

    separators = ["\n\n", "\n", ". ", " "]
    splits: list[str] = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        if end >= len(text):
            splits.append(text[start:])
            break

        # Find best split point
        split_point = end
        for sep in separators:
            idx = text.rfind(sep, start, end)
            if idx > start:
                split_point = idx + len(sep)
                break

        splits.append(text[start:split_point].strip())
        new_start = split_point - chunk_overlap
        if new_start <= start:
            new_start = start + 1
        start = new_start

    return [s for s in splits if s]
