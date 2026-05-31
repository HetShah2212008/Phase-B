"""
Pinecone vector store with Pinecone Inference embeddings.

Embeddings turn text into dense vectors so similar meaning → similar vectors.
At query time we embed the user's question and find the closest stored vectors
(cosine similarity) — that is the "Retrieval" in RAG.
"""

import logging
import os
import uuid
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Embedding model dimensions
# multilingual-e5-large produces 1024-dimensional dense vectors.
# ---------------------------------------------------------------------------
_EMBED_MODEL = "multilingual-e5-large"
_EMBED_DIM   = 1024
_BATCH_SIZE  = 96   # max inputs per Pinecone inference call

# ---------------------------------------------------------------------------
# Module-level lazy singletons
# ---------------------------------------------------------------------------
_pc_client = None   # pinecone.Pinecone instance
_pc_index  = None   # pinecone.Index instance


def _get_client():
    """Return (and lazily create) the Pinecone client."""
    global _pc_client
    if _pc_client is None:
        from pinecone import Pinecone  # deferred import — heavy at module level

        api_key = os.environ.get("PINECONE_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "PINECONE_API_KEY environment variable is not set. "
                "Add it to your .env file."
            )
        _pc_client = Pinecone(api_key=api_key)
        logger.info("Pinecone client initialised")
    return _pc_client


def _get_index():
    """Return (and lazily connect to) the Pinecone index, creating it if needed."""
    global _pc_index
    if _pc_index is None:
        from pinecone import ServerlessSpec  # deferred import

        pc = _get_client()
        index_name = os.environ.get("PINECONE_INDEX", "phaseb")

        existing = [idx.name for idx in pc.list_indexes()]
        if index_name not in existing:
            logger.info(
                "Pinecone index '%s' not found — creating (dim=%d, metric=cosine)",
                index_name, _EMBED_DIM,
            )
            pc.create_index(
                name=index_name,
                dimension=_EMBED_DIM,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            logger.info("Pinecone index '%s' created", index_name)
        else:
            logger.info("Pinecone index '%s' found", index_name)

        _pc_index = pc.Index(index_name)
    return _pc_index


def _embed(texts: list[str], *, input_type: str = "passage") -> list[list[float]]:
    """
    Embed a list of texts using Pinecone Inference (multilingual-e5-large).

    Args:
        texts:      Non-empty list of strings to embed.
        input_type: "passage" for documents being indexed,
                    "query"   for search queries.

    Returns:
        List of float vectors, one per input text.
    """
    pc = _get_client()
    all_vectors: list[list[float]] = []

    for i in range(0, len(texts), _BATCH_SIZE):
        batch = texts[i : i + _BATCH_SIZE]
        result = pc.inference.embed(
            model=_EMBED_MODEL,
            inputs=batch,
            parameters={"input_type": input_type, "truncate": "END"},
        )
        all_vectors.extend(emb.values for emb in result.data)

    return all_vectors


# ---------------------------------------------------------------------------
# VectorService — public interface
# ---------------------------------------------------------------------------

class VectorService:
    """
    Pinecone-backed vector store using multilingual-e5-large embeddings.

    All heavy initialisation (client, index connection, first embed call)
    is deferred to the first actual operation so startup RAM stays low.
    """

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def add_documents(
        self,
        documents: list[str],
        *,
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
    ) -> list[str]:
        """
        Embed and upsert text chunks into Pinecone.

        Args:
            documents:  List of text chunks to index.
            metadatas:  Optional per-chunk metadata dicts (must match length).
            ids:        Optional explicit IDs; UUIDs are generated if omitted.

        Returns:
            List of vector IDs stored in Pinecone.
        """
        if not documents:
            return []

        if metadatas is not None and len(metadatas) != len(documents):
            raise ValueError("metadatas length must match documents length")

        doc_ids   = ids or [str(uuid.uuid4()) for _ in documents]
        meta_list = metadatas or [{} for _ in documents]

        # Embed all chunks (passage mode)
        vectors = _embed(documents, input_type="passage")

        # Build upsert payload — store the original text in metadata so we
        # can return it from similarity_search without a separate fetch.
        upsert_payload = [
            {
                "id":       doc_id,
                "values":   vec,
                "metadata": {**meta, "_text": doc},
            }
            for doc_id, vec, meta, doc in zip(doc_ids, vectors, meta_list, documents)
        ]

        index = _get_index()
        index.upsert(vectors=upsert_payload, batch_size=_BATCH_SIZE)
        logger.info("Indexed %d chunk(s) in Pinecone", len(documents))
        return doc_ids

    def add_document_chunks(
        self,
        chunks: list[str],
        *,
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
    ) -> list[str]:
        """Alias for add_documents — keeps older call sites working."""
        return self.add_documents(chunks, metadatas=metadatas, ids=ids)

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def similarity_search(
        self,
        query: str,
        *,
        n_results: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve the top-k chunks most similar to the query.

        Retrieval flow:
          1. Embed the query with multilingual-e5-large (query mode)
          2. Pinecone compares query vector to all stored vectors (cosine)
          3. Return closest matches with score → converted to similarity

        Returns:
            List of dicts: id, document, metadata, distance, similarity.
            (distance here is 1 - cosine_score, for API compatibility.)
        """
        if not query.strip():
            return []

        from config import get_settings
        settings  = get_settings()
        top_k     = n_results if n_results is not None else settings.rag_top_k

        # Embed the query (query mode)
        query_vec = _embed([query], input_type="query")[0]

        index = _get_index()
        response = index.query(
            vector=query_vec,
            top_k=top_k,
            include_metadata=True,
        )

        hits: list[dict[str, Any]] = []
        for match in response.matches:
            score      = float(match.score)          # cosine similarity in [0, 1]
            distance   = max(0.0, 1.0 - score)       # distance for API compat
            raw_meta   = dict(match.metadata or {})
            text       = raw_meta.pop("_text", "")   # recover stored text

            hits.append(
                {
                    "id":         match.id,
                    "document":   text,
                    "metadata":   raw_meta,
                    "distance":   distance,
                    "similarity": score,
                }
            )

        logger.info("Retrieved %d chunk(s) for query", len(hits))
        return hits
