"""
Document loading and preprocessing utilities for the RAG pipeline.

Typical ingestion flow (API / future batch jobs):
  1. load_pdf()        → extract text per page
  2. split_documents() → break into overlapping chunks
  3. VectorService.add_documents() → embed and store in ChromaDB

The RAG Agent does NOT call this module — only upload/ingestion routes do.
"""

import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import get_settings

logger = logging.getLogger(__name__)


class DocumentService:
    """PDF loading, text extraction, and chunking for vector ingestion."""

    def __init__(self) -> None:
        settings = get_settings()
        # Chunk size / overlap come from config (defaults: 1000 / 200)
        self._chunk_size = settings.chunk_size
        self._chunk_overlap = settings.chunk_overlap
        self._upload_dir = Path(settings.upload_dir)
        self._upload_dir.mkdir(parents=True, exist_ok=True)

    @property
    def upload_dir(self) -> Path:
        return self._upload_dir

    def load_pdf(self, file_path: str | Path) -> list[Document]:
        """
        Load a PDF and return one LangChain Document per page.

        PyPDFLoader reads each page's text into `page_content`. Metadata
        (e.g. page number) is attached for later retrieval citations.

        Raises:
            FileNotFoundError: Path does not exist.
            ValueError: File is not a PDF or has no extractable text.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a PDF file, got: {path.suffix}")

        loader = PyPDFLoader(str(path))
        documents = loader.load()

        # Clean whitespace on each page so chunks are not polluted with noise
        cleaned: list[Document] = []
        for doc in documents:
            text = self.preprocess_text(doc.page_content)
            if not text:
                continue
            doc.page_content = text
            cleaned.append(doc)

        if not cleaned:
            raise ValueError(f"No extractable text found in PDF: {path.name}")

        logger.info("Loaded %d page(s) with text from %s", len(cleaned), path.name)
        return cleaned

    def split_documents(self, documents: list[Document]) -> list[Document]:
        """
        Split long documents into smaller chunks for embedding and retrieval.

        Why chunking is needed:
          LLMs and embedding models work on limited context windows. A full PDF
          is too large to embed as one vector — you would lose fine-grained
          matches when the user asks about one specific section.

        Why overlap matters (chunk_overlap=200):
          A sentence split across two chunks might lose meaning at the boundary.
          Overlap duplicates a few characters at edges so important context
          appears fully in at least one chunk → better recall during search.

        How chunking improves retrieval:
          Smaller, focused chunks produce embeddings that represent one idea.
          Similarity search then returns the exact paragraph related to the query
          instead of an entire 50-page document average.
        """
        if not documents:
            return []

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
            length_function=len,
            # Prefer natural breaks: paragraphs → lines → sentences → words
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        split_docs = splitter.split_documents(documents)

        # Drop empty chunks after splitting
        chunks = [doc for doc in split_docs if doc.page_content.strip()]
        logger.info(
            "Split %d source document(s) into %d chunk(s) "
            "(size=%d, overlap=%d)",
            len(documents),
            len(chunks),
            self._chunk_size,
            self._chunk_overlap,
        )
        return chunks

    def preprocess_text(self, text: str) -> str:
        """
        Normalize raw extracted text before chunking or display.

        PDF extractors often insert extra spaces and line breaks; collapsing
        whitespace keeps embeddings consistent with user queries.
        """
        return " ".join(text.split()).strip()

    def load_pdf_and_split(self, file_path: str | Path) -> list[Document]:
        """Convenience: load PDF then return chunked Document objects."""
        return self.split_documents(self.load_pdf(file_path))

    # Backward-compatible alias used during foundation phase
    def chunk_documents(self, documents: list[Document]) -> list[Document]:
        return self.split_documents(documents)
