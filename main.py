import io
import json

import pypdf
from fastapi import FastAPI, File, HTTPException, UploadFile

from agent import build_agent
from rag import get_retriever, ingest_text, is_loaded

app = FastAPI(title="Question Answering Service")


def extract_text_from_pdf(content: bytes) -> str:
    reader = pypdf.PdfReader(io.BytesIO(content))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def extract_text(filename: str, content: bytes) -> str:
    if filename.endswith(".json"):
        data = json.loads(content)
        return json.dumps(data, indent=2)
    elif filename.endswith(".pdf"):
        return extract_text_from_pdf(content)
    else:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Provide a .json or .pdf file.",
        )


@app.post("/ingest")
async def ingest(
    knowledge_file: UploadFile = File(..., description="PDF or JSON file to ingest into the RAG database"),
):
    content = await knowledge_file.read()
    text = extract_text(knowledge_file.filename, content)
    chunk_count = ingest_text(text)
    return {"message": "Knowledge base loaded successfully.", "chunks": chunk_count}


@app.post("/answer")
async def answer(
    questions_file: UploadFile = File(..., description="JSON file with a list of questions"),
    knowledge_file: UploadFile | None = File(default=None, description="Optional PDF or JSON file to load into the RAG database before answering"),
):
    if knowledge_file is not None:
        knowledge_content = await knowledge_file.read()
        text = extract_text(knowledge_file.filename, knowledge_content)
        ingest_text(text)
    elif not is_loaded():
        raise HTTPException(
            status_code=400,
            detail="No knowledge base loaded. Provide a knowledge_file or call POST /ingest first.",
        )

    questions_content = await questions_file.read()
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
