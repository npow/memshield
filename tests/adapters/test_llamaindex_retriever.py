"""Tests for memshield.adapters.llamaindex_retriever."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from memshield.adapters.llamaindex_retriever import Document, LlamaIndexRetrieverAdapter


def _make_node_with_score(text: str, metadata: dict, score: float) -> MagicMock:
    """Build a mock NodeWithScore."""
    node = MagicMock()
    node.get_content.return_value = text
    node.metadata = metadata
    nws = MagicMock()
    nws.node = node
    nws.score = score
    return nws


class TestLlamaIndexRetrieverAdapter:
    """Tests for LlamaIndexRetrieverAdapter."""

    def test_init_stores_retriever(self) -> None:
        """Constructor stores the retriever."""
        retriever = MagicMock()
        adapter = LlamaIndexRetrieverAdapter(retriever)
        assert adapter._retriever is retriever

    # ------------------------------------------------------------------
    # similarity_search
    # ------------------------------------------------------------------

    def test_similarity_search_returns_documents(self) -> None:
        """similarity_search converts NodeWithScore objects to Documents."""
        retriever = MagicMock()
        retriever.retrieve.return_value = [
            _make_node_with_score("hello world", {"source": "doc1"}, 0.9),
            _make_node_with_score("foo bar", {"source": "doc2"}, 0.7),
        ]
        adapter = LlamaIndexRetrieverAdapter(retriever)

        # Patch the import guard so the test doesn't need llama-index installed.
        with patch.object(adapter, "_check_import"):
            results = adapter.similarity_search("test query", k=4)

        assert len(results) == 2
        assert isinstance(results[0], Document)
        assert results[0].page_content == "hello world"
        assert results[0].metadata == {"source": "doc1"}
        assert results[1].page_content == "foo bar"

    def test_similarity_search_respects_k(self) -> None:
        """similarity_search truncates results to k items."""
        retriever = MagicMock()
        retriever.retrieve.return_value = [
            _make_node_with_score(f"text {i}", {}, 1.0 - i * 0.1) for i in range(10)
        ]
        adapter = LlamaIndexRetrieverAdapter(retriever)

        with patch.object(adapter, "_check_import"):
            results = adapter.similarity_search("q", k=3)

        assert len(results) == 3

    def test_similarity_search_calls_retrieve_with_query(self) -> None:
        """similarity_search forwards the query string to retriever.retrieve."""
        retriever = MagicMock()
        retriever.retrieve.return_value = []
        adapter = LlamaIndexRetrieverAdapter(retriever)

        with patch.object(adapter, "_check_import"):
            adapter.similarity_search("my special query", k=2)

        retriever.retrieve.assert_called_once_with("my special query")

    # ------------------------------------------------------------------
    # similarity_search_with_score
    # ------------------------------------------------------------------

    def test_similarity_search_with_score_returns_tuples(self) -> None:
        """similarity_search_with_score returns (Document, float) tuples."""
        retriever = MagicMock()
        retriever.retrieve.return_value = [
            _make_node_with_score("text A", {"k": "v"}, 0.88),
            _make_node_with_score("text B", {}, 0.55),
        ]
        adapter = LlamaIndexRetrieverAdapter(retriever)

        with patch.object(adapter, "_check_import"):
            results = adapter.similarity_search_with_score("q", k=4)

        assert len(results) == 2
        doc0, score0 = results[0]
        assert isinstance(doc0, Document)
        assert doc0.page_content == "text A"
        assert score0 == pytest.approx(0.88)
        _, score1 = results[1]
        assert score1 == pytest.approx(0.55)

    def test_similarity_search_with_score_respects_k(self) -> None:
        """similarity_search_with_score truncates results to k items."""
        retriever = MagicMock()
        retriever.retrieve.return_value = [
            _make_node_with_score(f"t{i}", {}, float(i)) for i in range(5)
        ]
        adapter = LlamaIndexRetrieverAdapter(retriever)

        with patch.object(adapter, "_check_import"):
            results = adapter.similarity_search_with_score("q", k=2)

        assert len(results) == 2

    def test_similarity_search_with_score_none_score(self) -> None:
        """A None score is converted to 0.0."""
        retriever = MagicMock()
        node_with_score = _make_node_with_score("text", {}, 0.0)
        node_with_score.score = None
        retriever.retrieve.return_value = [node_with_score]
        adapter = LlamaIndexRetrieverAdapter(retriever)

        with patch.object(adapter, "_check_import"):
            results = adapter.similarity_search_with_score("q", k=4)

        _, score = results[0]
        assert score == pytest.approx(0.0)

    # ------------------------------------------------------------------
    # ImportError
    # ------------------------------------------------------------------

    def test_import_error_message(self) -> None:
        """Helpful ImportError when llama-index-core is not installed."""
        retriever = MagicMock()
        retriever.retrieve.return_value = []
        adapter = LlamaIndexRetrieverAdapter(retriever)

        with patch.dict("sys.modules", {"llama_index": None, "llama_index.core": None, "llama_index.core.schema": None}):
            with pytest.raises(ImportError, match="pip install memshield\\[llamaindex\\]"):
                adapter._check_import()
