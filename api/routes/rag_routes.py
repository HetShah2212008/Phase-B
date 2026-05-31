"""
RAG HTTP endpoints: PDF ingestion and question answering.

Example upload (curl):
  curl -X POST "http://127.0.0.1:8000/upload-pdf" \\
    -H "accept: application/json" \\
    -H "Content-Type: multipart/form-data" \\
    -F "file=@./my_document.pdf"

Expected response:
  {
    "success": true,
    "filename": "my_document.pdf",
    "chunks_created": 42,
    "message": "PDF indexed successfully"
  }

Example query (curl):
  curl -X POST "http://127.0.0.1:8000/query" \\
    -H "Content-Type: application/json" \\
    -d '{"query": "What is the main topic of the document?"}'

Expected response:
  {
    "answer": "...",
    "retrieved_chunks": ["...", "..."],
    "sources": [
      {
        "text": "...",
        "metadata": {"source": "my_document.pdf", "page": 1},
        "similarity": 0.82,
        "distance": 0.18
      }
    ]
  }
"""

import logging
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from agents.rag_agent import RAGAgent
from api.dependencies import (
    get_document_service,
    get_rag_agent,
    get_vector_service,
)
from schemas.api_models import QueryRequest, QueryResponse, RetrievedChunkInfo, UploadPdfResponse
from schemas.pipeline_state import create_initial_state
from services.document_service import DocumentService
from services.vector_service import VectorService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["RAG"])


@router.post("/upload-pdf", response_model=UploadPdfResponse)
async def upload_pdf(
    file: UploadFile = File(..., description="PDF file to ingest into Pinecone"),
    document_service: DocumentService = Depends(get_document_service),
    vector_service: VectorService = Depends(get_vector_service),
) -> UploadPdfResponse:
    """
    Ingest a PDF: save → extract text → chunk → embed → store in Pinecone.

    This route owns document processing; the RAG Agent only queries the index.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Unique filename avoids collisions when uploading the same name twice
    safe_name = Path(file.filename).name
    stored_name = f"{uuid.uuid4().hex}_{safe_name}"
    dest_path = document_service.upload_dir / stored_name

    try:
        with dest_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 1) Extract text per page  2) Split into overlapping chunks
        pages = document_service.load_pdf(dest_path)
        chunks = document_service.split_documents(pages)

        texts = [doc.page_content for doc in chunks]
        metadatas = []
        for i, doc in enumerate(chunks):
            meta = dict(doc.metadata or {})
            meta.update(
                {
                    "source": safe_name,
                    "stored_filename": stored_name,
                    "chunk_index": i,
                }
            )
            metadatas.append(meta)

        # 3) Embed with multilingual-e5-large and persist in Pinecone
        vector_service.add_documents(texts, metadatas=metadatas)

        return UploadPdfResponse(
            success=True,
            filename=safe_name,
            chunks_created=len(texts),
            message="PDF indexed successfully",
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("PDF upload failed")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc


@router.post("/query", response_model=QueryResponse)
async def query_documents(
    body: QueryRequest,
    rag_agent: RAGAgent = Depends(get_rag_agent),
) -> QueryResponse:
    """
    Run the RAG agent: retrieve chunks → generate grounded Gemini answer.

    RAGAgent absorbs ALL Gemini errors internally and returns a fallback
    response — this route will never see a Gemini-caused exception.
    The only exceptions that reach here are genuine infrastructure failures
    (e.g. Pinecone unreachable), which are surfaced as HTTP 500.
    """
    state = create_initial_state(query=body.query)

    try:
        updated = await rag_agent.run(state)
    except ValueError as exc:
        # Bad input (empty query, etc.)
        logger.warning("[/query] rejected (400): %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        # True infrastructure failure (Pinecone, etc.) — Gemini errors never reach here.
        logger.exception("[/query] infrastructure failure: %s", exc)
        raise HTTPException(
            status_code=500, detail="An unexpected infrastructure error occurred."
        ) from exc

    used_fallback = bool(updated.get("used_fallback_response", False))
    if used_fallback:
        logger.info("[/query] returning fallback response (Gemini unavailable)")

    details = updated.get("retrieval_details") or []
    sources = [
        RetrievedChunkInfo(
            text=hit.get("document", ""),
            metadata=hit.get("metadata") or {},
            similarity=float(hit.get("similarity", 0.0)),
            distance=hit.get("distance"),
        )
        for hit in details
    ]

    return QueryResponse(
        answer=updated.get("rag_response") or "",
        retrieved_chunks=updated.get("retrieved_chunks") or [],
        sources=sources,
        used_fallback_response=used_fallback,
    )


@router.get("/forecast", tags=["Forecasting"])
async def get_cashflow_forecast(
    starting_balance: float = 500000,
    monthly_burn: float = 45000,
    months: int = 12,
    forecast_periods: int = 12,
):
    """
    Run the cash flow forecast with configurable parameters.

    Query params:
        starting_balance:  Initial cash balance (default 500000)
        monthly_burn:      Monthly spend rate (default 45000)
        months:            Months of historical data to generate (default 12)
        forecast_periods:  Months ahead to forecast (default 12)

    Returns:
        status, forecast list, zero_date, chart_path
    """
    import os

    from ml.forecasting import generate_synthetic_cashflow, run_cashflow_forecast

    output_path = os.path.join("uploads", "cashflow_forecast.png")
    os.makedirs("uploads", exist_ok=True)

    historical_df = generate_synthetic_cashflow(
        starting_balance=starting_balance,
        monthly_burn=monthly_burn,
        months=months,
    )
    result = run_cashflow_forecast(
        historical_df=historical_df,
        forecast_periods=forecast_periods,
        output_path=output_path,
    )
    return result


@router.get("/forecast/chart", tags=["Forecasting"])
async def get_forecast_chart():
    """
    Return the cash flow forecast chart as a PNG image.

    Generates the chart first if it does not already exist.
    """
    import os

    from fastapi.responses import FileResponse

    from ml.forecasting import run_cashflow_forecast

    output_path = os.path.join("uploads", "cashflow_forecast.png")
    os.makedirs("uploads", exist_ok=True)

    if not os.path.exists(output_path):
        run_cashflow_forecast(output_path=output_path)

    return FileResponse(output_path, media_type="image/png")
