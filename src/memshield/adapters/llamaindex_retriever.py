"""LlamaIndex retriever adapter for MemShield."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    """Minimal LangChain-compatible document with page content and metadata."""

    page_content: str
    metadata: dict = field(default_factory=dict)


class LlamaIndexRetrieverAdapter:
    """Wraps a LlamaIndex retriever to be compatible with MemShield.wrap().

    LlamaIndex retrievers expose ``retriever.retrieve(query)`` which returns a
    list of ``NodeWithScore`` objects.  This adapter converts that interface to
    the ``similarity_search`` / ``similarity_search_with_score`` contract that
    ``VectorStoreProxy`` expects.

    Usage::

        from memshield.adapters.llamaindex_retriever import LlamaIndexRetrieverAdapter
        adapted = LlamaIndexRetrieverAdapter(your_index.as_retriever())
        store = shield.wrap(adapted)
        docs = store.similarity_search("query")
    """

    def __init__(self, retriever: Any) -> None:
        """Initialise the adapter with a LlamaIndex retriever.

        Args:
            retriever: A LlamaIndex retriever that exposes a ``retrieve``
                method returning ``List[NodeWithScore]``.
        """
        self._retriever = retriever

    def _check_import(self) -> None:
        """Verify that llama-index-core is installed."""
        try:
            from llama_index.core.schema import NodeWithScore  # noqa: F401
        except ImportError:
            raise ImportError(
                "llama-index-core is required. "
                "Install with: pip install memshield[llamaindex]"
            ) from None

    def _nodes_to_documents(self, nodes: list[Any]) -> list[Document]:
        """Convert a list of NodeWithScore objects to Document objects.

        Args:
            nodes: List of ``NodeWithScore`` instances.

        Returns:
            A list of :class:`Document` objects.
        """
        docs: list[Document] = []
        for node_with_score in nodes:
            text = node_with_score.node.get_content()
            metadata = dict(node_with_score.node.metadata or {})
            docs.append(Document(page_content=text, metadata=metadata))
        return docs

    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> list[Document]:
        """Return the top-k documents most similar to *query*.

        Args:
            query: The search query string.
            k: Number of results to return.
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            A list of :class:`Document` objects ordered by relevance.
        """
        self._check_import()
        nodes = self._retriever.retrieve(query)
        docs = self._nodes_to_documents(nodes)
        return docs[:k]

    def similarity_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[tuple[Document, float]]:
        """Return the top-k documents paired with their relevance scores.

        Args:
            query: The search query string.
            k: Number of results to return.
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            A list of ``(Document, score)`` tuples ordered by relevance.
        """
        self._check_import()
        nodes = self._retriever.retrieve(query)
        results: list[tuple[Document, float]] = []
        for node_with_score in nodes[:k]:
            text = node_with_score.node.get_content()
            metadata = dict(node_with_score.node.metadata or {})
            doc = Document(page_content=text, metadata=metadata)
            score = float(node_with_score.score or 0.0)
            results.append((doc, score))
        return results
