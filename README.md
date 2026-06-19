# AgentQuestionAnswers

A FastAPI service that answers questions from a knowledge base using Retrieval-Augmented Generation (RAG). Upload a PDF or JSON knowledge file, ask questions, and receive answers grounded strictly in the provided content — powered by LangGraph and OpenAI.

---

## Table of Contents

- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Setup](#setup)
- [Running the Service](#running-the-service)
- [API Reference](#api-reference)
- [Usage Examples](#usage-examples)
- [Input File Formats](#input-file-formats)
- [Test Suite](#test-suite)

---

## How It Works

```
Knowledge File (PDF or JSON)
        │
        ▼
  Text Extraction
        │
        ▼
  Chunking (RecursiveCharacterTextSplitter)
        │
        ▼
  OpenAI Embeddings (text-embedding-3-large) → InMemoryVectorStore (RAG database)
        │
        ▼
  Question File (JSON list of strings)
        │
        ▼
  LangGraph Agent
    ├── For each question → similarity search → retrieve top-k relevant chunks
    └── OpenAI LLM (gpt-4o-mini) → answer strictly from retrieved context
        │
        ▼
  [{"question": "...", "answer": "..."}, ...]
```

### Key design decisions

- **RAG over full-context**: Instead of passing the entire document to the LLM, the agent retrieves only the most relevant chunks per question. This keeps prompts focused and scales to larger documents.
- **Strict grounding**: The LLM is instructed to answer only from the retrieved context. If the answer cannot be found, it returns `"I cannot find the answer from given source"`.
- **In-memory store**: The vector store lives in process memory. It resets when the server restarts. For persistence across restarts, swap `InMemoryVectorStore` for a persistent backend (e.g. ChromaDB, Pinecone).
- **LangGraph orchestration**: The agent is a compiled `StateGraph`. Each node is a discrete step, making it straightforward to add retrieval re-ranking, multi-hop reasoning, or other nodes in the future.

---

## Project Structure

```
AgentQuestionAnswers/
├── main.py              # FastAPI app — /ingest and /answer endpoints
├── agent.py             # LangGraph agent — retrieves context and answers questions
├── rag.py               # In-memory RAG store — chunking, embedding, retrieval
├── requirements.txt     # Python dependencies
├── .env                 # API keys (not committed)
├── docs/
│   ├── sample.json      # Example knowledge base (Q&A format)
│   ├── questions.json   # Example questions file
│   └── soc2-type2.pdf   # Example knowledge base (PDF)
└── tests/
    ├── test_main.py     # Tests for FastAPI endpoints and text extraction
    ├── test_agent.py    # Tests for LangGraph agent logic
    └── test_rag.py      # Tests for RAG store (ingest, retrieval, lifecycle)
```

---

## Requirements

- Python 3.11+
- An [OpenAI API key](https://platform.openai.com/api-keys)

### Python packages

| Package | Purpose |
|---|---|
| `fastapi` | REST API framework |
| `uvicorn` | ASGI server |
| `python-multipart` | File upload support for FastAPI |
| `langgraph` | Agent orchestration as a state graph |
| `langchain-openai` | OpenAI LLM (`gpt-4o-mini`) and embeddings (`text-embedding-3-large`) via LangChain |
| `langchain` | Core LangChain utilities |
| `langchain-text-splitters` | Document chunking |
| `pypdf` | PDF text extraction |
| `python-dotenv` | Load environment variables from `.env` |

---

## Setup

**1. Clone the repository**

```bash
git clone <repo-url>
cd AgentQuestionAnswers
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Configure environment variables**

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your-openai-api-key-here
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-large
```

`OPENAI_MODEL` is optional and defaults to `gpt-4o-mini` if omitted.
`OPENAI_EMBEDDING_MODEL` is optional and defaults to `text-embedding-3-large` if omitted.

> The `.env` file is listed in `.gitignore` and will not be committed.

---

## Running the Service

```bash
python3 -m uvicorn main:app --reload
```

The service starts at `http://localhost:8000`.

Interactive API docs (Swagger UI) are available at `http://localhost:8000/docs`.

---

## API Reference

### `POST /ingest`

Load a knowledge file into the RAG database. Call this once before querying, or use the `knowledge_file` parameter on `/answer` to ingest and query in a single request.

**Request** — multipart form data

| Field | Type | Required | Description |
|---|---|---|---|
| `knowledge_file` | File | Yes | PDF or JSON file to ingest |

**Response**

```json
{
  "message": "Knowledge base loaded successfully.",
  "chunks": 42
}
```

`chunks` is the number of text segments stored in the vector database.

---

### `POST /answer`

Answer a list of questions using the RAG database.

**Request** — multipart form data

| Field | Type | Required | Description |
|---|---|---|---|
| `questions_file` | File | Yes | JSON file containing a list of question strings |
| `knowledge_file` | File | No | PDF or JSON file — if provided, ingested into RAG before answering |

If `knowledge_file` is omitted, the service uses the knowledge base already loaded via `/ingest`. Returns `400` if no knowledge base is available.

**Response**

```json
[
  {
    "question": "Where are your data centres located?",
    "answer": "US Central region, hosted on Google Cloud Platform."
  },
  {
    "question": "Do you use quantum encryption?",
    "answer": "I cannot find the answer from given source"
  }
]
```

**Error responses**

| Status | Reason |
|---|---|
| `400` | `questions_file` is not a JSON array of strings |
| `400` | No knowledge base loaded and no `knowledge_file` provided |
| `400` | `knowledge_file` is not a `.pdf` or `.json` file |

---

## Usage Examples

### Two-step: ingest then query

```bash
# Step 1 — load the knowledge base
curl -X POST http://localhost:8000/ingest \
  -F "knowledge_file=@docs/sample.json"

# Step 2 — ask questions
curl -X POST http://localhost:8000/answer \
  -F "questions_file=@docs/questions.json"
```

### One-step: ingest and query together

```bash
curl -X POST http://localhost:8000/answer \
  -F "questions_file=@docs/questions.json" \
  -F "knowledge_file=@docs/soc2-type2.pdf"
```

---

## Input File Formats

### Questions file

A JSON array of strings. Each string is a question.

```json
[
  "Where are your data centres located?",
  "Do you monitor and restrict the installation of unauthorized software?",
  "Is there a dedicated sanctions compliance officer?"
]
```

### Knowledge file — JSON

Any valid JSON structure. The full content is serialised to a string, chunked, and embedded. A natural format is a list of Q&A objects:

```json
[
  {
    "question": "Where are your data centres located?",
    "answer": "US Central region on GCP.",
    "confidence": "high"
  }
]
```

### Knowledge file — PDF

Any standard PDF. Text is extracted page by page using `pypdf` and then chunked.

---

## Test Suite

Tests are written with `pytest` and use `unittest.mock` to avoid calling OpenAI APIs.

```bash
python3 -m pytest tests/ -v
```

### Test coverage

**`tests/test_rag.py`** — RAG store lifecycle

| Test | What it verifies |
|---|---|
| `test_ingest_returns_chunk_count` | `ingest_text` returns the number of chunks created |
| `test_ingest_calls_add_texts` | Text chunks are added to the vector store |
| `test_ingest_sets_store` | `is_loaded()` is `True` after ingestion |
| `test_ingest_overwrites_previous_store` | Re-ingesting replaces the existing store |
| `test_get_retriever_raises_when_not_loaded` | `get_retriever()` raises before any ingestion |
| `test_get_retriever_uses_default_k` | Default retrieval top-k is 4 |
| `test_get_retriever_custom_k` | Custom `k` is passed through to the retriever |
| `test_is_loaded_false_initially` | Store starts empty |
| `test_is_loaded_true_after_ingest` | Store is marked loaded after ingestion |
| `test_clear_resets_store` | `clear()` unloads the store |
| `test_clear_when_already_empty` | `clear()` is safe to call on an empty store |

**`tests/test_agent.py`** — LangGraph agent behaviour

| Test | What it verifies |
|---|---|
| `test_returns_one_answer_per_question` | Each question produces exactly one answer |
| `test_multiple_questions` | All questions answered in order |
| `test_strips_whitespace_from_answer` | LLM response whitespace is trimmed |
| `test_empty_questions` | Empty input returns empty list, LLM not called |
| `test_cannot_find_answer_passthrough` | Fallback text passed through unchanged |
| `test_retriever_called_once_per_question` | Retriever is invoked once per question |
| `test_system_prompt_passed_to_llm` | System prompt is always included |
| `test_retrieved_context_included_in_prompt` | Retrieved chunks appear in the user message |
| `test_build_agent_returns_compiled_graph` | `build_agent()` returns a non-null graph |

**`tests/test_main.py`** — FastAPI endpoints and text extraction

| Test | What it verifies |
|---|---|
| `test_extract_text_json` | JSON file is parsed and re-serialised |
| `test_extract_text_unsupported_format` | `.txt` raises HTTP 400 |
| `test_extract_text_from_pdf_returns_string` | PDF bytes yield a string |
| `test_extract_text_pdf` | End-to-end PDF extraction via `extract_text` |
| `test_ingest_json_success` | `/ingest` with JSON returns chunk count |
| `test_ingest_pdf_success` | `/ingest` with PDF succeeds |
| `test_ingest_unsupported_format` | `/ingest` with `.txt` returns 400 |
| `test_answer_returns_list` | `/answer` returns a list of Q&A objects |
| `test_answer_without_ingest_returns_400` | `/answer` with no knowledge loaded returns 400 |
| `test_answer_with_knowledge_file_ingests_then_answers` | `knowledge_file` triggers ingestion before answering |
| `test_answer_with_knowledge_file_does_not_require_prior_ingest` | `knowledge_file` works without a prior `/ingest` call |
| `test_answer_invalid_questions_not_a_list` | Non-array questions file returns 400 |
| `test_answer_questions_not_strings` | Array of non-strings returns 400 |
| `test_answer_agent_invoked_with_correct_state` | Agent receives the correct state shape |
