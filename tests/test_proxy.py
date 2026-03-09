"""Tests for memshield.proxy."""
from __future__ import annotations

from unittest.mock import MagicMock

from memshield.proxy import VectorStoreProxy


class TestVectorStoreProxy:
    """Tests for VectorStoreProxy."""

    def test_delegates_unknown_attributes(self) -> None:
        """Attributes not intercepted pass through to wrapped store."""
        store = MagicMock()
        store.some_method.return_value = "result"
        proxy = VectorStoreProxy(store, shield=MagicMock())
        assert proxy.some_method() == "result"
        store.some_method.assert_called_once()

    def test_delegates_properties(self) -> None:
        """Properties pass through to wrapped store."""
        store = MagicMock()
        store.collection_name = "test_collection"
        proxy = VectorStoreProxy(store, shield=MagicMock())
        assert proxy.collection_name == "test_collection"

    def test_repr(self) -> None:
        """Repr shows proxy and wrapped type."""
        store = MagicMock()
        proxy = VectorStoreProxy(store, shield=MagicMock())
        assert "VectorStoreProxy" in repr(proxy)
        assert "MagicMock" in repr(proxy)

    def test_similarity_search_calls_validate(self) -> None:
        """similarity_search passes results through shield.validate_reads."""
        store = MagicMock()
        doc1 = MagicMock(page_content="safe content")
        doc2 = MagicMock(page_content="poisoned content")
        store.similarity_search.return_value = [doc1, doc2]

        shield = MagicMock()
        shield.audit_log = None  # no audit path
        shield.validate_reads.return_value = [doc1]  # filters out doc2

        proxy = VectorStoreProxy(store, shield=shield)
        results = proxy.similarity_search("query", k=2)

        store.similarity_search.assert_called_once_with("query", k=2)
        shield.validate_reads.assert_called_once_with([doc1, doc2])
        assert results == [doc1]

    def test_similarity_search_with_score(self) -> None:
        """similarity_search_with_score filters by validated docs."""
        store = MagicMock()
        doc1 = MagicMock(page_content="safe")
        doc2 = MagicMock(page_content="poisoned")
        store.similarity_search_with_score.return_value = [(doc1, 0.9), (doc2, 0.8)]

        shield = MagicMock()
        shield.audit_log = None  # no audit path
        shield.validate_reads.return_value = [doc1]

        proxy = VectorStoreProxy(store, shield=shield)
        results = proxy.similarity_search_with_score("query", k=2)

        assert len(results) == 1
        assert results[0][0] is doc1
        assert results[0][1] == 0.9

    def test_add_documents_calls_provenance(self) -> None:
        """add_documents tags provenance before forwarding to store."""
        store = MagicMock()
        store.add_documents.return_value = ["id1"]
        shield = MagicMock()

        docs = [MagicMock(page_content="new entry")]
        proxy = VectorStoreProxy(store, shield=shield)
        result = proxy.add_documents(docs)

        shield.tag_provenance.assert_called_once_with(docs)
        store.add_documents.assert_called_once_with(docs)
        assert result == ["id1"]

    def test_add_texts_calls_provenance(self) -> None:
        """add_texts tags provenance before forwarding to store."""
        store = MagicMock()
        store.add_texts.return_value = ["id1"]
        shield = MagicMock()

        proxy = VectorStoreProxy(store, shield=shield)
        result = proxy.add_texts(["hello"], metadatas=[{"source": "user"}])

        shield.tag_provenance_texts.assert_called_once_with(["hello"], [{"source": "user"}])
        store.add_texts.assert_called_once_with(["hello"], metadatas=[{"source": "user"}])
        assert result == ["id1"]

    def test_kwargs_forwarded(self) -> None:
        """Extra kwargs are forwarded to the wrapped store."""
        store = MagicMock()
        store.similarity_search.return_value = []
        shield = MagicMock()
        shield.audit_log = None  # no audit path
        shield.validate_reads.return_value = []

        proxy = VectorStoreProxy(store, shield=shield)
        proxy.similarity_search("query", k=10, filter={"type": "doc"})

        store.similarity_search.assert_called_once_with("query", k=10, filter={"type": "doc"})
