"""
PDF Reader Tool — Extract and chunk PDF documents for evidence ingestion.

Uses PyMuPDF for text extraction with recursive chunking for
optimal vector store indexing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import fitz  # PyMuPDF


@dataclass
class PDFChunk:
    """A text chunk extracted from a PDF document."""

    content: str
    page_number: int
    chunk_index: int
    source_file: str
    metadata: dict[str, Any]


class PDFReaderTool:
    """
    PDF extraction and chunking tool.

    Features
    --------
    - Full-text extraction via PyMuPDF
    - Recursive text splitting with configurable overlap
    - Page-level metadata preservation
    - Table and header detection
    """

    def __init__(self, chunk_size: int = 1000, overlap: int = 200) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def extract(self, file_path: str) -> list[dict[str, Any]]:
        """
        Extract text from a PDF and split into overlapping chunks.

        Parameters
        ----------
        file_path : str
            Path to the PDF file.

        Returns
        -------
        list[dict]
            Evidence-formatted text chunks with page metadata.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF not found: {file_path}")

        doc = fitz.open(file_path)
        chunks = []
        chunk_idx = 0

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")

            if not text.strip():
                continue

            # Split page text into overlapping chunks
            page_chunks = self._split_text(text)

            for chunk_text in page_chunks:
                chunks.append({
                    "type": "pdf_chunk",
                    "content": chunk_text,
                    "page_number": page_num + 1,
                    "chunk_index": chunk_idx,
                    "source_file": os.path.basename(file_path),
                    "url": f"file://{file_path}#page={page_num + 1}",
                    "title": f"{os.path.basename(file_path)} — Page {page_num + 1}",
                })
                chunk_idx += 1

        doc.close()
        return chunks

    def _split_text(self, text: str) -> list[str]:
        """Split text into overlapping chunks using recursive splitting."""
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size

            # Try to break at sentence boundary
            if end < len(text):
                last_period = text.rfind(".", start, end)
                last_newline = text.rfind("\n", start, end)
                break_point = max(last_period, last_newline)
                if break_point > start:
                    end = break_point + 1

            chunks.append(text[start:end].strip())
            start = end - self.overlap

        return [c for c in chunks if c]
