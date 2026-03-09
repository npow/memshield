"""pgvector (psycopg2) vector store adapter for MemShield."""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Document:
    """Minimal LangChain-compatible document with page content and metadata."""

    page_content: str
    metadata: dict = field(default_factory=dict)


class PgVectorStoreAdapter:
    """Wraps a pgvector table to be compatible with MemShield.wrap().

    The table must have the following columns (names are configurable):

    * ``id``        — primary key (uuid or serial)
    * ``content``   — text chunk (text)
    * ``embedding`` — vector column (vector)
    * ``metadata``  — arbitrary metadata (jsonb)

    Similarity is computed with the ``<->`` (L2) operator provided by the
    pgvector extension.

    Usage::

        from memshield.adapters.pgvector_store import PgVectorStoreAdapter
        adapter = PgVectorStoreAdapter(
            dsn="postgresql://user:pass@localhost/db",
            table="documents",
            embed_fn=my_embed_fn,
        )
        store = shield.wrap(adapter)
        docs = store.similarity_search("query text")
    """

    def __init__(
        self,
        dsn: str,
        table: str,
        embed_fn: Callable[[str], list[float]],
        id_column: str = "id",
        content_column: str = "content",
        embedding_column: str = "embedding",
        metadata_column: str = "metadata",
    ) -> None:
        """Initialise the adapter.

        Args:
            dsn: libpq-compatible connection string, e.g.
                ``"postgresql://user:pass@host/dbname"``.
            table: Name of the table that stores embeddings.
            embed_fn: Callable that converts a text string to an embedding
                vector (list of floats).
            id_column: Name of the primary-key column.
            content_column: Name of the text content column.
            embedding_column: Name of the pgvector column.
            metadata_column: Name of the JSONB metadata column.
        """
        self._dsn = dsn
        self._table = table
        self._embed_fn = embed_fn
        self._id_column = id_column
        self._content_column = content_column
        self._embedding_column = embedding_column
        self._metadata_column = metadata_column

    def _check_import(self) -> None:
        """Verify that psycopg2 is installed."""
        try:
            import psycopg2  # noqa: F401
        except ImportError:
            raise ImportError(
                "psycopg2 is required. "
                "Install with: pip install memshield[pgvector]"
            ) from None

    def _get_connection(self) -> Any:
        """Open and return a psycopg2 connection, registering pgvector types.

        Returns:
            An open psycopg2 connection object.
        """
        import psycopg2  # type: ignore[import]

        conn = psycopg2.connect(self._dsn)
        # Register the pgvector type adapter if the package is available.
        try:
            from pgvector.psycopg2 import register_vector  # type: ignore[import]

            register_vector(conn)
        except ImportError:
            pass
        return conn

    def _vector_literal(self, vector: list[float]) -> str:
        """Format a Python list as a pgvector literal string.

        Args:
            vector: The embedding vector.

        Returns:
            A string in the form ``"[0.1,0.2,...]"`` suitable for casting
            with ``::vector`` in SQL.
        """
        return "[" + ",".join(str(v) for v in vector) + "]"

    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> list[Document]:
        """Return the top-k documents most similar to *query*.

        Uses the pgvector ``<->`` (L2 distance) operator.

        Args:
            query: The search query string.
            k: Number of results to return.
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            A list of :class:`Document` objects ordered by ascending distance.
        """
        self._check_import()
        vector = self._embed_fn(query)
        vec_literal = self._vector_literal(vector)
        sql = (
            f"SELECT {self._content_column}, {self._metadata_column} "
            f"FROM {self._table} "
            f"ORDER BY {self._embedding_column} <-> %s::vector "
            f"LIMIT %s"
        )
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, (vec_literal, k))
                rows = cur.fetchall()
        finally:
            conn.close()

        docs: list[Document] = []
        for content, meta in rows:
            if isinstance(meta, str):
                meta = json.loads(meta)
            docs.append(Document(page_content=content, metadata=meta or {}))
        return docs

    def similarity_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[tuple[Document, float]]:
        """Return the top-k documents paired with their L2 distance scores.

        Lower scores indicate greater similarity.

        Args:
            query: The search query string.
            k: Number of results to return.
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            A list of ``(Document, distance)`` tuples.
        """
        self._check_import()
        vector = self._embed_fn(query)
        vec_literal = self._vector_literal(vector)
        sql = (
            f"SELECT {self._content_column}, {self._metadata_column}, "
            f"{self._embedding_column} <-> %s::vector AS distance "
            f"FROM {self._table} "
            f"ORDER BY distance "
            f"LIMIT %s"
        )
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, (vec_literal, k))
                rows = cur.fetchall()
        finally:
            conn.close()

        results: list[tuple[Document, float]] = []
        for content, meta, distance in rows:
            if isinstance(meta, str):
                meta = json.loads(meta)
            doc = Document(page_content=content, metadata=meta or {})
            results.append((doc, float(distance)))
        return results

    def add_documents(self, documents: list[Any], **kwargs: Any) -> list[str]:
        """Insert documents into the pgvector table.

        Each document must have a ``page_content`` attribute (str) and
        optionally a ``metadata`` attribute (dict).

        Args:
            documents: List of document-like objects with ``page_content`` and
                ``metadata`` attributes.
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            A list of UUID strings for the newly inserted rows.
        """
        self._check_import()
        conn = self._get_connection()
        ids: list[str] = []
        try:
            with conn.cursor() as cur:
                for doc in documents:
                    doc_id = str(uuid.uuid4())
                    content = doc.page_content
                    metadata = getattr(doc, "metadata", {}) or {}
                    vector = self._embed_fn(content)
                    vec_literal = self._vector_literal(vector)
                    sql = (
                        f"INSERT INTO {self._table} "
                        f"({self._id_column}, {self._content_column}, "
                        f"{self._embedding_column}, {self._metadata_column}) "
                        f"VALUES (%s, %s, %s::vector, %s)"
                    )
                    cur.execute(
                        sql,
                        (doc_id, content, vec_literal, json.dumps(metadata)),
                    )
                    ids.append(doc_id)
            conn.commit()
        finally:
            conn.close()
        return ids
