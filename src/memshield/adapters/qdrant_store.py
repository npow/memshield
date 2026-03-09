"""Qdrant vector store adapter for MemShield."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Document:
    """Minimal LangChain-compatible document with page content and metadata."""

    page_content: str
    metadata: dict = field(default_factory=dict)


class QdrantStoreAdapter:
    """Wraps a Qdrant collection to be compatible with MemShield.wrap().

    Uses the ``qdrant-client`` Python SDK.  The adapter maps Qdrant
    ``ScoredPoint`` objects (from ``client.search``) to :class:`Document`
    objects with ``.page_content`` and ``.metadata``.

    Usage::

        from qdrant_client import QdrantClient
        from memshield.adapters.qdrant_store import QdrantStoreAdapter

        adapter = QdrantStoreAdapter(
            client=QdrantClient(url="http://localhost:6333"),
            collection_name="my_collection",
            embed_fn=my_embed_fn,
        )
        store = shield.wrap(adapter)
        docs = store.similarity_search("query text")
    """

    def __init__(
        self,
        client: Any,
        collection_name: str,
        embed_fn: Callable[[str], list[float]],
        text_key: str = "text",
    ) -> None:
        """Initialise the adapter.

        Args:
            client: An initialised ``qdrant_client.QdrantClient`` instance.
            collection_name: Name of the Qdrant collection to search / upsert.
            embed_fn: Callable that converts a text string to an embedding
                vector (list of floats).
            text_key: The payload key that holds the raw chunk text.
        """
        self._client = client
        self._collection_name = collection_name
        self._embed_fn = embed_fn
        self._text_key = text_key

    def _check_import(self) -> None:
        """Verify that qdrant-client is installed."""
        try:
            from qdrant_client import QdrantClient  # noqa: F401
        except ImportError:
            raise ImportError(
                "qdrant-client is required. "
                "Install with: pip install memshield[qdrant]"
            ) from None

    def _scored_points_to_documents(self, points: list[Any]) -> list[Document]:
        """Convert ScoredPoint objects to Document objects.

        Args:
            points: List of ``qdrant_client.models.ScoredPoint`` instances.

        Returns:
            A list of :class:`Document` objects.
        """
        docs: list[Document] = []
        for point in points:
            payload = dict(point.payload or {})
            text = payload.pop(self._text_key, "")
            docs.append(Document(page_content=text, metadata=payload))
        return docs

    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> list[Document]:
        """Return the top-k documents most similar to *query*.

        Args:
            query: The search query string.
            k: Number of results to return.
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            A list of :class:`Document` objects ordered by descending score.
        """
        self._check_import()
        vector = self._embed_fn(query)
        points = self._client.search(
            collection_name=self._collection_name,
            query_vector=vector,
            limit=k,
            with_payload=True,
        )
        return self._scored_points_to_documents(points)

    def similarity_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[tuple[Document, float]]:
        """Return the top-k documents paired with their similarity scores.

        Args:
            query: The search query string.
            k: Number of results to return.
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            A list of ``(Document, score)`` tuples ordered by descending score.
        """
        self._check_import()
        vector = self._embed_fn(query)
        points = self._client.search(
            collection_name=self._collection_name,
            query_vector=vector,
            limit=k,
            with_payload=True,
        )
        results: list[tuple[Document, float]] = []
        for point in points:
            payload = dict(point.payload or {})
            text = payload.pop(self._text_key, "")
            doc = Document(page_content=text, metadata=payload)
            results.append((doc, float(point.score)))
        return results

    def add_documents(self, documents: list[Any], **kwargs: Any) -> list[str]:
        """Upsert documents into the Qdrant collection.

        Each document must have a ``page_content`` attribute (str) and
        optionally a ``metadata`` attribute (dict).

        Args:
            documents: List of document-like objects with ``page_content`` and
                ``metadata`` attributes.
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            A list of UUID strings used as point IDs.
        """
        self._check_import()
        try:
            from qdrant_client.models import PointStruct  # type: ignore[import]
        except ImportError:
            raise ImportError(
                "qdrant-client is required. "
                "Install with: pip install memshield[qdrant]"
            ) from None

        ids: list[str] = []
        points: list[Any] = []
        for doc in documents:
            doc_id = str(uuid.uuid4())
            content = doc.page_content
            metadata = dict(getattr(doc, "metadata", {}) or {})
            vector = self._embed_fn(content)
            payload = {self._text_key: content, **metadata}
            points.append(PointStruct(id=doc_id, vector=vector, payload=payload))
            ids.append(doc_id)

        self._client.upsert(
            collection_name=self._collection_name,
            points=points,
        )
        return ids
