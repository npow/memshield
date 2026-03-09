"""Tests for memshield.adapters.pgvector_store."""
from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from memshield.adapters.pgvector_store import Document, PgVectorStoreAdapter


def _dummy_embed(text: str) -> list[float]:
    """Return a fixed embedding vector for testing."""
    return [0.1, 0.2, 0.3]


class TestPgVectorStoreAdapter:
    """Tests for PgVectorStoreAdapter."""

    def test_init_stores_params(self) -> None:
        """Constructor stores all configuration values."""
        adapter = PgVectorStoreAdapter(
            dsn="postgresql://user:pass@localhost/db",
            table="docs",
            embed_fn=_dummy_embed,
            content_column="body",
            metadata_column="meta",
        )
        assert adapter._dsn == "postgresql://user:pass@localhost/db"
        assert adapter._table == "docs"
        assert adapter._content_column == "body"
        assert adapter._metadata_column == "meta"

    # ------------------------------------------------------------------
    # similarity_search
    # ------------------------------------------------------------------

    def test_similarity_search_returns_documents(self) -> None:
        """similarity_search converts DB rows to Documents."""
        adapter = PgVectorStoreAdapter(
            dsn="dsn", table="documents", embed_fn=_dummy_embed
        )

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchall.return_value = [
            ("chunk one", {"source": "file1"}),
            ("chunk two", {}),
        ]
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)

        with patch.object(adapter, "_check_import"):
            with patch.object(adapter, "_get_connection", return_value=mock_conn):
                results = adapter.similarity_search("query", k=4)

        assert len(results) == 2
        assert isinstance(results[0], Document)
        assert results[0].page_content == "chunk one"
        assert results[0].metadata == {"source": "file1"}
        assert results[1].page_content == "chunk two"

    def test_similarity_search_executes_correct_sql(self) -> None:
        """similarity_search runs a SELECT with ORDER BY distance."""
        adapter = PgVectorStoreAdapter(
            dsn="dsn", table="documents", embed_fn=_dummy_embed
        )
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchall.return_value = []
        mock_conn.cursor.return_value = mock_cursor

        with patch.object(adapter, "_check_import"):
            with patch.object(adapter, "_get_connection", return_value=mock_conn):
                adapter.similarity_search("test", k=3)

        sql_call = mock_cursor.execute.call_args
        sql_text = sql_call[0][0]
        assert "SELECT" in sql_text
        assert "ORDER BY" in sql_text
        assert "<->" in sql_text
        # k is passed as second param
        assert sql_call[0][1][1] == 3

    def test_similarity_search_parses_json_string_metadata(self) -> None:
        """Metadata stored as JSON string is parsed to dict."""
        adapter = PgVectorStoreAdapter(
            dsn="dsn", table="documents", embed_fn=_dummy_embed
        )
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchall.return_value = [('text', '{"key": "val"}')]
        mock_conn.cursor.return_value = mock_cursor

        with patch.object(adapter, "_check_import"):
            with patch.object(adapter, "_get_connection", return_value=mock_conn):
                results = adapter.similarity_search("q", k=1)

        assert results[0].metadata == {"key": "val"}

    # ------------------------------------------------------------------
    # similarity_search_with_score
    # ------------------------------------------------------------------

    def test_similarity_search_with_score_returns_tuples(self) -> None:
        """similarity_search_with_score returns (Document, float) tuples."""
        adapter = PgVectorStoreAdapter(
            dsn="dsn", table="documents", embed_fn=_dummy_embed
        )
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchall.return_value = [
            ("text A", {"src": "a"}, 0.12),
            ("text B", {}, 0.45),
        ]
        mock_conn.cursor.return_value = mock_cursor

        with patch.object(adapter, "_check_import"):
            with patch.object(adapter, "_get_connection", return_value=mock_conn):
                results = adapter.similarity_search_with_score("q", k=4)

        assert len(results) == 2
        doc0, score0 = results[0]
        assert isinstance(doc0, Document)
        assert doc0.page_content == "text A"
        assert score0 == pytest.approx(0.12)

    def test_similarity_search_with_score_sql_includes_distance(self) -> None:
        """similarity_search_with_score SQL selects the distance expression."""
        adapter = PgVectorStoreAdapter(
            dsn="dsn", table="documents", embed_fn=_dummy_embed
        )
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchall.return_value = []
        mock_conn.cursor.return_value = mock_cursor

        with patch.object(adapter, "_check_import"):
            with patch.object(adapter, "_get_connection", return_value=mock_conn):
                adapter.similarity_search_with_score("q", k=2)

        sql_text = mock_cursor.execute.call_args[0][0]
        assert "<->" in sql_text
        assert "distance" in sql_text.lower() or "AS" in sql_text

    # ------------------------------------------------------------------
    # add_documents
    # ------------------------------------------------------------------

    def test_add_documents_inserts_rows(self) -> None:
        """add_documents inserts each document and returns UUIDs."""
        import uuid as uuid_mod

        adapter = PgVectorStoreAdapter(
            dsn="dsn", table="documents", embed_fn=_dummy_embed
        )
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cursor

        docs = [
            Document(page_content="doc one", metadata={"a": 1}),
            Document(page_content="doc two", metadata={}),
        ]

        with patch.object(adapter, "_check_import"):
            with patch.object(adapter, "_get_connection", return_value=mock_conn):
                ids = adapter.add_documents(docs)

        assert len(ids) == 2
        for id_ in ids:
            uuid_mod.UUID(id_)  # raises if not valid UUID

    def test_add_documents_calls_execute_for_each(self) -> None:
        """add_documents calls cursor.execute once per document."""
        adapter = PgVectorStoreAdapter(
            dsn="dsn", table="documents", embed_fn=_dummy_embed
        )
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cursor

        docs = [Document(page_content=f"doc {i}") for i in range(3)]

        with patch.object(adapter, "_check_import"):
            with patch.object(adapter, "_get_connection", return_value=mock_conn):
                adapter.add_documents(docs)

        assert mock_cursor.execute.call_count == 3

    def test_add_documents_commits(self) -> None:
        """add_documents commits the transaction."""
        adapter = PgVectorStoreAdapter(
            dsn="dsn", table="documents", embed_fn=_dummy_embed
        )
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cursor

        with patch.object(adapter, "_check_import"):
            with patch.object(adapter, "_get_connection", return_value=mock_conn):
                adapter.add_documents([Document(page_content="x")])

        mock_conn.commit.assert_called_once()

    # ------------------------------------------------------------------
    # ImportError
    # ------------------------------------------------------------------

    def test_import_error_message(self) -> None:
        """Helpful ImportError when psycopg2 is not installed."""
        adapter = PgVectorStoreAdapter(
            dsn="dsn", table="documents", embed_fn=_dummy_embed
        )

        with patch.dict("sys.modules", {"psycopg2": None}):
            with pytest.raises(ImportError, match="pip install memshield\\[pgvector\\]"):
                adapter._check_import()
