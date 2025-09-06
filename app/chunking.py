import re
from typing import List


def chunk_by_paragraphs(text: str, max_chars: int = 1200) -> List[str]:
    if max_chars <= 0:
        max_chars = 1200
    # Split paragraphs by blank lines
    raw_paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0
    for para in raw_paragraphs:
        extra = (2 if current else 0) + len(para)
        if current_len + extra <= max_chars:
            current.append(para)
            current_len += extra
        else:
            if current:
                chunks.append("\n\n".join(current))
            if len(para) <= max_chars:
                current = [para]
                current_len = len(para)
            else:
                # Paragraph longer than max_chars: hard-split it
                for start in range(0, len(para), max_chars):
                    chunks.append(para[start : start + max_chars])
                current = []
                current_len = 0
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def chunk_by_characters(text: str, chunk_size: int = 1000, overlap: int = 120) -> List[str]:
    if chunk_size <= 0:
        chunk_size = 1000
    if overlap < 0:
        overlap = 0
    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 4)

    chunks: List[str] = []
    start = 0
    text_len = len(text)
    step = max(1, chunk_size - overlap)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunks.append(text[start:end])
        if end == text_len:
            break
        start += step
    return chunks

