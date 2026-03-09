"""Pinecone vector store adapter for MemShield."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Document:
    """Minimal LangChain-compatible document with page content and metadata."""

    page_content: str
    metadata: dict = field(default_factory=dict)


class PineconeStoreAdapter:
    """Wraps a Pinecone Index to be compatible with MemShield.wrap().

    Requires an embedding function to convert query strings to vectors because
    the Pinecone client operates at the vector level, not the text level.

    Usage::

        from memshield.adapters.pinecone_store import PineconeStoreAdapter
        adapter = PineconeStoreAdapter(index, embed_fn=my_embed_fn, namespace="default")
        store = shield.wrap(adapter)
        docs = store.similarity_search("query text")
    """

    def __init__(
        self,
        index: Any,
        embed_fn: Callable[[str], list[float]],
        namespace: str = "",
        text_key: str = "text",
    ) -> None:
        """Initialise the adapter.

        Args:
            index: A ``pinecone.Index`` instance.
            embed_fn: A callable that converts a query string into an embedding
                vector (list of floats).
            namespace: Pinecone namespace to query and upsert into.
            text_key: The metadata key in each Pinecone match that holds the
                raw chunk text.
        """
        self._index = index
        self._embed_fn = embed_fn
        self._namespace = namespace
        self._text_key = text_key

    def _check_import(self) -> None:
        """Verify that pinecone-client is installed."""
        try:
            import pinecone  # noqa: F401
        except ImportError:
            raise ImportError(
                "pinecone-client is required. "
                "Install with: pip install memshield[pinecone]"
            ) from None

    def _matches_to_documents(self, matches: list[Any]) -> list[Document]:
        """Convert Pinecone query matches to Document objects.

        Args:
            matches: List of Pinecone ``ScoredVector`` / match objects.

        Returns:
            A list of :class:`Document` objects.
        """
        docs: list[Document] = []
        for match in matches:
            metadata = dict(match.metadata or {})
            text = metadata.pop(self._text_key, "")
            docs.append(Document(page_content=text, metadata=metadata))
        return docs

    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> list[Document]:
        """Return the top-k documents most similar to *query*.

        Embeds the query with ``embed_fn``, calls ``index.query``, and converts
        the results to :class:`Document` objects.

        Args:
            query: The search query string.
            k: Number of results to return.
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            A list of :class:`Document` objects.
        """
        self._check_import()
        vector = self._embed_fn(query)
        response = self._index.query(
            vector=vector,
            top_k=k,
            namespace=self._namespace,
            include_metadata=True,
        )
        return self._matches_to_documents(response.matches)

    def similarity_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[tuple[Document, float]]:
        """Return the top-k documents paired with their similarity scores.

        Args:
            query: The search query string.
            k: Number of results to return.
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            A list of ``(Document, score)`` tuples.
        """
        self._check_import()
        vector = self._embed_fn(query)
        response = self._index.query(
            vector=vector,
            top_k=k,
            namespace=self._namespace,
            include_metadata=True,
        )
        results: list[tuple[Document, float]] = []
        for match in response.matches:
            metadata = dict(match.metadata or {})
            text = metadata.pop(self._text_key, "")
            doc = Document(page_content=text, metadata=metadata)
            results.append((doc, float(match.score)))
        return results
