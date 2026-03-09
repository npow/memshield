"""Tests for memshield.adapters.pinecone_store."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from memshield.adapters.pinecone_store import Document, PineconeStoreAdapter


def _dummy_embed(text: str) -> list[float]:
    """Return a fixed embedding vector for testing."""
    return [0.1, 0.2, 0.3]


def _make_match(text: str, extra_meta: dict, score: float) -> MagicMock:
    """Build a mock Pinecone query match."""
    match = MagicMock()
    match.metadata = {"text": text, **extra_meta}
    match.score = score
    return match


class TestPineconeStoreAdapter:
    """Tests for PineconeStoreAdapter."""

    def test_init_stores_params(self) -> None:
        """Constructor stores index, embed_fn, namespace, and text_key."""
        index = MagicMock()
        adapter = PineconeStoreAdapter(
            index, embed_fn=_dummy_embed, namespace="ns", text_key="content"
        )
        assert adapter._index is index
        assert adapter._embed_fn is _dummy_embed
        assert adapter._namespace == "ns"
        assert adapter._text_key == "content"

    # ------------------------------------------------------------------
    # similarity_search
    # ------------------------------------------------------------------

    def test_similarity_search_returns_documents(self) -> None:
        """similarity_search converts Pinecone matches to Documents."""
        index = MagicMock()
        index.query.return_value = MagicMock(
            matches=[
                _make_match("hello world", {"source": "s1"}, 0.95),
                _make_match("foo bar", {}, 0.80),
            ]
        )
        adapter = PineconeStoreAdapter(index, embed_fn=_dummy_embed)

        with patch.object(adapter, "_check_import"):
            results = adapter.similarity_search("query", k=4)

        assert len(results) == 2
        assert isinstance(results[0], Document)
        assert results[0].page_content == "hello world"
        assert results[0].metadata == {"source": "s1"}
        assert results[1].page_content == "foo bar"

    def test_similarity_search_calls_index_query(self) -> None:
        """similarity_search passes correct arguments to index.query."""
        index = MagicMock()
        index.query.return_value = MagicMock(matches=[])
        adapter = PineconeStoreAdapter(index, embed_fn=_dummy_embed, namespace="ns1")

        with patch.object(adapter, "_check_import"):
            adapter.similarity_search("test", k=5)

        index.query.assert_called_once_with(
            vector=[0.1, 0.2, 0.3],
            top_k=5,
            namespace="ns1",
            include_metadata=True,
        )

    def test_similarity_search_text_key_removed_from_metadata(self) -> None:
        """The text_key entry should not appear in Document.metadata."""
        index = MagicMock()
        index.query.return_value = MagicMock(
            matches=[_make_match("content text", {"extra": "yes"}, 0.5)]
        )
        adapter = PineconeStoreAdapter(index, embed_fn=_dummy_embed, text_key="text")

        with patch.object(adapter, "_check_import"):
            results = adapter.similarity_search("q", k=1)

        assert "text" not in results[0].metadata
        assert results[0].metadata == {"extra": "yes"}

    # ------------------------------------------------------------------
    # similarity_search_with_score
    # ------------------------------------------------------------------

    def test_similarity_search_with_score_returns_tuples(self) -> None:
        """similarity_search_with_score returns (Document, float) tuples."""
        index = MagicMock()
        index.query.return_value = MagicMock(
            matches=[
                _make_match("text A", {}, 0.92),
                _make_match("text B", {}, 0.71),
            ]
        )
        adapter = PineconeStoreAdapter(index, embed_fn=_dummy_embed)

        with patch.object(adapter, "_check_import"):
            results = adapter.similarity_search_with_score("q", k=4)

        assert len(results) == 2
        doc0, score0 = results[0]
        assert isinstance(doc0, Document)
        assert doc0.page_content == "text A"
        assert score0 == pytest.approx(0.92)

    def test_similarity_search_with_score_float_cast(self) -> None:
        """Scores are cast to float."""
        index = MagicMock()
        match = _make_match("t", {}, 1)  # integer score
        index.query.return_value = MagicMock(matches=[match])
        adapter = PineconeStoreAdapter(index, embed_fn=_dummy_embed)

        with patch.object(adapter, "_check_import"):
            results = adapter.similarity_search_with_score("q", k=1)

        _, score = results[0]
        assert isinstance(score, float)

    # ------------------------------------------------------------------
    # ImportError
    # ------------------------------------------------------------------

    def test_import_error_message(self) -> None:
        """Helpful ImportError when pinecone-client is not installed."""
        index = MagicMock()
        adapter = PineconeStoreAdapter(index, embed_fn=_dummy_embed)

        with patch.dict("sys.modules", {"pinecone": None}):
            with pytest.raises(ImportError, match="pip install memshield\\[pinecone\\]"):
                adapter._check_import()
