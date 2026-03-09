"""VectorStoreProxy — intercepts similarity_search and add_documents."""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class VectorStoreProxy:
    """Transparent proxy that intercepts vector store reads and writes.

    Reads go through validation. Writes get provenance tags.
    All other methods delegate to the wrapped store.
    """

    def __init__(self, wrapped: Any, shield: Any) -> None:
        object.__setattr__(self, "_wrapped", wrapped)
        object.__setattr__(self, "_shield", shield)

    def __getattr__(self, name: str) -> Any:
        """Delegate all unknown attributes to the wrapped store."""
        return getattr(self._wrapped, name)

    def __repr__(self) -> str:
        return f"VectorStoreProxy(wrapping={type(self._wrapped).__name__})"

    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> list[Any]:
        """Intercept reads: validate results before returning to the agent."""
        inference_id = kwargs.pop("inference_id", None)
        user_id = kwargs.pop("user_id", None)

        results = self._wrapped.similarity_search(query, k=k, **kwargs)

        if self._shield.audit_log is not None:
            # Attempt to get scores for audit logging without a double query;
            # use id(doc) as key to map scores back to docs from results list.
            try:
                scored = self._wrapped.similarity_search_with_score(query, k=k, **kwargs)
                score_map = {id(doc): score for doc, score in scored}
            except (AttributeError, NotImplementedError):
                score_map = {}

            clean_docs, _ = self._shield.validate_reads_with_audit(
                results,
                query=query,
                user_id=user_id,
                inference_id=inference_id,
                score_map=score_map,
            )
            return clean_docs

        return self._shield.validate_reads(results)

    def similarity_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[tuple[Any, float]]:
        """Intercept scored reads: validate results before returning."""
        inference_id = kwargs.pop("inference_id", None)
        user_id = kwargs.pop("user_id", None)

        results = self._wrapped.similarity_search_with_score(query, k=k, **kwargs)
        docs = [doc for doc, _score in results]
        score_map = {id(doc): score for doc, score in results}

        if self._shield.audit_log is not None:
            clean_docs, _ = self._shield.validate_reads_with_audit(
                docs,
                query=query,
                user_id=user_id,
                inference_id=inference_id,
                score_map=score_map,
            )
        else:
            clean_docs = self._shield.validate_reads(docs)

        clean_set = {id(d) for d in clean_docs}
        return [(doc, score) for doc, score in results if id(doc) in clean_set]

    def add_documents(self, documents: list[Any], **kwargs: Any) -> list[str]:
        """Intercept writes: tag with provenance before storing."""
        self._shield.tag_provenance(documents)
        return self._wrapped.add_documents(documents, **kwargs)

    def add_texts(self, texts: list[str], metadatas: list[dict[str, Any]] | None = None, **kwargs: Any) -> list[str]:
        """Intercept text writes: tag with provenance before storing."""
        self._shield.tag_provenance_texts(texts, metadatas)
        return self._wrapped.add_texts(texts, metadatas=metadatas, **kwargs)
