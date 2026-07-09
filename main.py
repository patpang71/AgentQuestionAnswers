import io
import json
import os

import pypdf
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile

from agent import build_agent
from rag import get_ingestion_status, get_retriever, ingest_background, ingest_text, is_loaded

app = FastAPI(title="Question Answering Service")

MAX_UPLOAD_SIZE_BYTES = int(os.getenv("MAX_UPLOAD_SIZE_BYTES", str(20 * 1024 * 1024)))
_UPLOAD_READ_CHUNK_BYTES = 1024 * 1024


async def read_upload(file: UploadFile, max_bytes: int = MAX_UPLOAD_SIZE_BYTES) -> bytes:
    """Read an UploadFile in bounded chunks, rejecting it before it's fully
    buffered in memory if it exceeds max_bytes."""
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(_UPLOAD_READ_CHUNK_BYTES)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds maximum allowed size of {max_bytes} bytes.",
            )
        chunks.append(chunk)
    return b"".join(chunks)


def extract_text_from_pdf(content: bytes) -> str:
    try:
        reader = pypdf.PdfReader(io.BytesIO(content))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not parse PDF file: {exc}",
        )


def extract_text(filename: str, content: bytes) -> str:
    if filename.endswith(".json"):
        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid JSON in knowledge_file: {exc}",
            )
        return json.dumps(data, indent=2)
    elif filename.endswith(".pdf"):
        return extract_text_from_pdf(content)
    else:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Provide a .json or .pdf file.",
        )


@app.post("/ingest", status_code=202)
async def ingest(
    background_tasks: BackgroundTasks,
    knowledge_file: UploadFile = File(..., description="PDF or JSON file to ingest into the RAG database"),
):
    content = await read_upload(knowledge_file)
    text = extract_text(knowledge_file.filename, content)
    background_tasks.add_task(ingest_background, text)
    return {"message": "Ingestion started. Poll GET /status to check progress."}


@app.get("/status")
async def status():
    return get_ingestion_status()


@app.post("/answer")
async def answer(
    questions_file: UploadFile = File(..., description="JSON file with a list of questions"),
    knowledge_file: UploadFile | None = File(default=None, description="Optional PDF or JSON file to load into the RAG database before answering"),
):
    if knowledge_file is not None:
        knowledge_content = await read_upload(knowledge_file)
        text = extract_text(knowledge_file.filename, knowledge_content)
        ingest_text(text)
    else:
        rag_status = get_ingestion_status()
        if rag_status["status"] == "processing":
            raise HTTPException(
                status_code=409,
                detail="Knowledge base is still being ingested. Try again shortly or poll GET /status.",
            )
        elif not is_loaded():
            raise HTTPException(
                status_code=400,
                detail="No knowledge base loaded. Provide a knowledge_file or call POST /ingest first.",
            )

    questions_content = await read_upload(questions_file)
    try:
        questions: list[str] = json.loads(questions_content)
        if not isinstance(questions, list) or not all(isinstance(q, str) for q in questions):
            raise ValueError
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="questions_file must be a JSON array of strings.",
        )

    retriever = get_retriever()
    agent = build_agent(retriever)
    result = agent.invoke({"questions": questions, "answers": []})

    return result["answers"]
