# AI Agent Pipeline — Part 1 Backend Foundation

College project backend using **FastAPI**, **LangChain**, **LangGraph**, **Google Gemini**, **Pinecone**, **Prophet**, and **Gmail SMTP**.

Part 1 includes a **working RAG pipeline** (PDF ingest, vector retrieval, grounded Gemini answers). Content/Email agents and Part 2 (disaster system) are not implemented yet.

---

## Project structure

```
backend/
├── agents/              # One module per agent (RAG, Content, Email)
├── pipelines/           # LangGraph and Google ADK orchestration
├── services/            # Shared infrastructure (Gemini, Pinecone, email, PDFs)
├── ml/                  # Prophet time-series forecasting
├── api/                 # FastAPI routes and dependency injection
├── schemas/             # PipelineState + API request/response models
├── data/                # SQLite (later)
├── uploads/             # Incoming PDFs for RAG
├── tests/               # Pytest tests
├── utils/               # Small shared helpers
├── config.py            # Environment variables (pydantic-settings)
├── main.py              # FastAPI app
├── requirements.txt
└── .env.example         # Copy to .env and fill in secrets
```

### How modules work together

| Module | Role |
|--------|------|
| **schemas/pipeline_state.py** | Single shared state object passed between agents |
| **services/** | Reusable capabilities (no orchestration logic) |
| **agents/** | Business steps: each agent reads/writes specific state keys |
| **pipelines/** | Wires agents into a graph (LangGraph) or sequence (ADK stub) |
| **ml/forecasting.py** | Optional Prophet step; writes `forecast_data` into state |

---

## RAG API (implemented)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/upload-pdf` | POST | Upload PDF → chunk → embed → Pinecone |
| `/query` | POST | Run RAG agent → answer + retrieved chunks |

**Ingest a document**

```bash
curl -X POST "http://127.0.0.1:8000/upload-pdf" \
  -F "file=@./your_document.pdf"
```

**Ask a question**

```bash
curl -X POST "http://127.0.0.1:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the document about?"}'
```

Requires `GEMINI_API_KEY` and `PINECONE_API_KEY` in `.env`.

---

## Future Part 1 flow

1. **User** sends a query (and optional email recipient) via API.
2. **RAG Agent** searches Pinecone for relevant chunks, then asks Gemini for a grounded answer → `rag_response`.
3. **Content Agent** formats the answer (and optional forecast) into a report → `formatted_content`.
4. **Email Agent** sends the report via SMTP → `email_status`.
5. **Forecasting** (when wired) runs on time-series input → `forecast_data` before or after content formatting.

```
User Query
    │
    ▼
┌─────────┐     ┌──────────────┐     ┌─────────────┐
│ RAG     │ ──► │ Content      │ ──► │ Email       │
│ Agent   │     │ Agent        │     │ Agent       │
└─────────┘     └──────────────┘     └─────────────┘
    │                  ▲
    │                  │ (optional)
    ▼                  │
 Pinecone          Forecasting
 + Gemini              (Prophet)
```

---

## LangGraph orchestration concept

**LangGraph** treats your pipeline as a **graph**:

- **Nodes** = functions (our agents' `run` methods).
- **Edges** = order of execution (`rag` → `content` → `email`).
- **State** = one dictionary (`PipelineState`) updated at each step.

Each node receives the full state, updates only its fields, and returns the state for the next node. This makes it easy to add Part 2 branches (e.g. "if disaster detected, go to alert node") without rewriting RAG or Email agents.

See `pipelines/langgraph_pipeline.py` for the stub graph definition.

---

## How agents communicate

Agents **do not call each other directly**. They communicate only through **shared state**:

```python
{
  "query": "",
  "retrieved_chunks": [],
  "rag_response": "",
  "formatted_content": "",
  "email_recipient": "",
  "email_status": "",
  "forecast_data": {}
}
```

- **RAG Agent** reads `query`, writes `retrieved_chunks` and `rag_response`.
- **Content Agent** reads `rag_response` (and optionally `forecast_data`), writes `formatted_content`.
- **Email Agent** reads `formatted_content` and `email_recipient`, writes `email_status`.

Services (`GeminiService`, `VectorService`, etc.) are injected into agents for testability.

---

## Google ADK pipeline

`pipelines/adk_pipeline.py` provides a **parallel orchestration path** using the same agents and state. When you integrate Google ADK, you will register agents with ADK's runner instead of LangGraph's `StateGraph`. The foundation keeps both pipelines aligned so behavior stays consistent.

---

## Setup

```bash
cd backend
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
copy .env.example .env   # Windows
# cp .env.example .env   # macOS/Linux
```

Edit `.env` with your `GEMINI_API_KEY`, `PINECONE_API_KEY`, and SMTP settings.

```bash
uvicorn main:app --reload
```

- API docs: http://127.0.0.1:8000/docs  
- Health: http://127.0.0.1:8000/health  

---

## Part 2 expansion (not built yet)

The layout supports adding:

- New state fields in `schemas/pipeline_state.py`
- New agents under `agents/`
- New LangGraph nodes and conditional edges in `langgraph_pipeline.py`
- SQLite persistence via `SQLITE_DB_PATH`

Do not implement Part 2 disaster logic until Part 1 is complete.

---

## Testing

```bash
cd backend
pytest tests/ -v
```

---

## License / course use

Built for academic demonstration. Keep API keys in `.env` only — never commit `.env`.
