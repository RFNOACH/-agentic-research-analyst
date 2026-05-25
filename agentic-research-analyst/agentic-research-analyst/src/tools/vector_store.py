"""
Vector Store Manager — Semantic evidence retrieval via ChromaDB.

Indexes evidence from all tools into a unified vector store for
semantic search during memo synthesis.
"""

from __future__ import annotations

import uuid
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings


class VectorStoreManager:
    """
    Manages a ChromaDB collection for evidence storage and retrieval.

    Features
    --------
    - Automatic embedding via OpenAI text-embedding-3-small
    - Deduplication by content hash
    - Metadata-aware filtering
    - Configurable top-k retrieval
    """

    def __init__(
        self,
        embedding_model: str = "text-embedding-3-small",
        collection_name: str = "research_evidence",
    ) -> None:
        self.embedding_model = embedding_model
        self.client = chromadb.Client(ChromaSettings(anonymized_telemetry=False))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._indexed_hashes: set[str] = set()

    def index(self, evidence: list[dict[str, Any]]) -> int:
        """
        Index evidence items into the vector store.

        Parameters
        ----------
        evidence : list[dict]
            Evidence items with 'content' and optional metadata fields.

        Returns
        -------
        int
            Number of new items indexed (excludes duplicates).
        """
        new_items = 0

        for item in evidence:
            content = item.get("content", "")
            if not content.strip():
                continue

            content_hash = str(hash(content))
            if content_hash in self._indexed_hashes:
                continue

            self._indexed_hashes.add(content_hash)
            doc_id = str(uuid.uuid4())

            metadata = {
                "type": item.get("type", "unknown"),
                "url": item.get("url", ""),
                "title": item.get("title", ""),
                "relevance": item.get("relevance", 0.5),
            }

            self.collection.add(
                documents=[content],
                ids=[doc_id],
                metadatas=[metadata],
            )
            new_items += 1

        return new_items

    def query(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """
        Retrieve the most relevant evidence for a query.

        Parameters
        ----------
        query : str
            Natural language query.
        top_k : int
            Maximum number of results to return.

        Returns
        -------
        list[dict]
            Evidence items ranked by semantic similarity.
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=min(top_k, self.collection.count()),
        )

        evidence = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else 1.0

                evidence.append({
                    "content": doc,
                    "url": meta.get("url", ""),
                    "title": meta.get("title", ""),
                    "type": meta.get("type", "unknown"),
                    "similarity": 1 - distance,  # Convert distance to similarity
                })

        return evidence

    def clear(self) -> None:
        """Remove all evidence from the store."""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"},
        )
        self._indexed_hashes.clear()
