"""
ChromaDB vector store with sentence-transformers embeddings.

Embeddings turn text into dense vectors so similar meaning → similar vectors.
At query time we embed the user's question and find the closest stored chunks
(cosine similarity in HNSW index) — that is the "Retrieval" in RAG.
"""

import logging
import uuid
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.utils import embedding_functions

from config import get_settings

logger = logging.getLogger(__name__)

_embedding_model = None


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        settings = get_settings()
        _embedding_model = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=settings.embedding_model,
        )
    return _embedding_model


class VectorService:
    """
    Persistent Chroma collection using all-MiniLM-L6-v2 embeddings.

    Storage layout per chunk:
      - documents: chunk text
      - metadatas: source file, page, chunk index, etc.
      - embeddings: computed by SentenceTransformer (not stored manually)
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._collection_name = settings.chroma_collection_name
        self._embedding_model = settings.embedding_model
        self._persist_dir = settings.chroma_persist_dir
        self._chroma_client: chromadb.PersistentClient | None = None
        self._collection: Collection | None = None
        # Both the ChromaDB client and embedding model are loaded lazily on first use

    @property
    def _client(self) -> chromadb.PersistentClient:
        if self._chroma_client is None:
            self._chroma_client = chromadb.PersistentClient(path=self._persist_dir)
        return self._chroma_client

    def get_or_create_collection(self, name: str | None = None) -> Collection:
        """
        Return a collection configured with our embedding function.

        `metadata={"hnsw:space": "cosine"}` tells Chroma to rank by cosine
        distance — standard for normalized sentence embeddings.
        """
        collection_name = name or self._collection_name
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=get_embedding_model(),
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "Chroma collection '%s' ready (embeddings=%s)",
            collection_name,
            self._embedding_model,
        )
        return self._collection

    @property
    def collection(self) -> Collection:
        if self._collection is None:
            return self.get_or_create_collection()
        return self._collection

    def add_documents(
        self,
        documents: list[str],
        *,
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
    ) -> list[str]:
        """
        Embed and store text chunks in ChromaDB.

        Chroma calls the SentenceTransformer embedding function internally:
          text → 384-dim vector → stored in HNSW index with metadata.

        Returns:
            List of chunk IDs assigned in the database.
        """
        if not documents:
            return []

        collection = self.collection
        doc_ids = ids or [str(uuid.uuid4()) for _ in documents]
        meta_list = metadatas or [{} for _ in documents]

        if len(meta_list) != len(documents):
            raise ValueError("metadatas length must match documents length")

        collection.add(
            documents=documents,
            metadatas=meta_list,
            ids=doc_ids,
        )
        logger.info("Indexed %d chunk(s) in '%s'", len(documents), collection.name)
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

    def similarity_search(
        self,
        query: str,
        *,
        n_results: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve the top-k chunks most similar to the query (dynamic retrieval).

        Retrieval flow:
          1. Embed the query with the same MiniLM model used at ingest time
          2. Chroma compares query vector to all chunk vectors (cosine distance)
          3. Return closest chunks with distance → converted to similarity score

        Returns:
            List of dicts: id, document, metadata, distance, similarity.
        """
        if not query.strip():
            return []

        settings = get_settings()
        top_k = n_results if n_results is not None else settings.rag_top_k
        collection = self.collection

        # Cap results by how many documents exist (Chroma errors if k > count)
        count = collection.count()
        if count == 0:
            logger.warning("Similarity search on empty collection")
            return []
        top_k = min(top_k, count)

        results = collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        hits: list[dict[str, Any]] = []
        for doc_id, doc, meta, dist in zip(ids, documents, metadatas, distances):
            # Cosine distance: 0 = identical, 2 = opposite. Map to similarity in [0,1].
            similarity = self._distance_to_similarity(dist)
            hits.append(
                {
                    "id": doc_id,
                    "document": doc,
                    "metadata": meta or {},
                    "distance": dist,
                    "similarity": similarity,
                }
            )

        logger.info("Retrieved %d chunk(s) for query", len(hits))
        return hits

    @staticmethod
    def _distance_to_similarity(distance: float | None) -> float:
        """Convert Chroma cosine distance to a human-readable similarity score."""
        if distance is None:
            return 0.0
        # Clamp for numerical safety
        return max(0.0, min(1.0, 1.0 - float(distance)))
