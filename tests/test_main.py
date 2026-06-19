import json
import zlib
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

import rag
from main import app, extract_text, extract_text_from_pdf

client = TestClient(app)

SAMPLE_QUESTIONS = ["Where are your data centres located?", "Do you have MFA?"]
SAMPLE_KNOWLEDGE = json.dumps([
    {"question": "Where are your data centres located?", "answer": "US Central on GCP."},
])
MOCK_ANSWERS = [
    {"question": "Where are your data centres located?", "answer": "US Central on GCP."},
    {"question": "Do you have MFA?", "answer": "Yes, MFA is enforced."},
]


@pytest.fixture(autouse=True)
def reset_rag():
    rag.clear()
    yield
    rag.clear()


def minimal_pdf() -> bytes:
    stream = b"BT /F1 12 Tf 100 700 Td (Hello) Tj ET"
    compressed = zlib.compress(stream)
    objects = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        3: b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        4: f"<< /Length {len(compressed)} /Filter /FlateDecode >>\nstream\n".encode() + compressed + b"\nendstream",
        5: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    }
    body = b"%PDF-1.4\n"
    offsets = {}
    for num, obj_body in objects.items():
        offsets[num] = len(body)
        body += f"{num} 0 obj\n".encode() + obj_body + b"\nendobj\n"
    xref_offset = len(body)
    body += b"xref\n" + f"0 {len(objects) + 1}\n".encode()
    body += b"0000000000 65535 f \n"
    for num in range(1, len(objects) + 1):
        body += f"{offsets[num]:010d} 00000 n \n".encode()
    body += (
        b"trailer\n"
        + f"<< /Size {len(objects) + 1} /Root 1 0 R >>\n".encode()
        + b"startxref\n"
        + f"{xref_offset}\n".encode()
        + b"%%EOF\n"
    )
    return body


# --- extract_text ---

def test_extract_text_json():
    result = extract_text("knowledge.json", SAMPLE_KNOWLEDGE.encode())
    assert json.loads(result) == json.loads(SAMPLE_KNOWLEDGE)


def test_extract_text_unsupported_format():
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc_info:
        extract_text("knowledge.txt", b"some content")
    assert exc_info.value.status_code == 400


def test_extract_text_from_pdf_returns_string():
    result = extract_text_from_pdf(minimal_pdf())
    assert isinstance(result, str)


def test_extract_text_pdf():
    result = extract_text("doc.pdf", minimal_pdf())
    assert isinstance(result, str)


# --- POST /ingest ---

@patch("main.ingest_background")
def test_ingest_returns_202(mock_ingest_bg):
    response = client.post(
        "/ingest",
        files={"knowledge_file": ("knowledge.json", SAMPLE_KNOWLEDGE.encode(), "application/json")},
    )
    assert response.status_code == 202
    assert "started" in response.json()["message"].lower()


@patch("main.ingest_background")
def test_ingest_triggers_background_task(mock_ingest_bg):
    client.post(
        "/ingest",
        files={"knowledge_file": ("knowledge.json", SAMPLE_KNOWLEDGE.encode(), "application/json")},
    )
    mock_ingest_bg.assert_called_once()


@patch("main.ingest_background")
def test_ingest_pdf_returns_202(mock_ingest_bg):
    response = client.post(
        "/ingest",
        files={"knowledge_file": ("doc.pdf", minimal_pdf(), "application/pdf")},
    )
    assert response.status_code == 202


def test_ingest_unsupported_format():
    response = client.post(
        "/ingest",
        files={"knowledge_file": ("doc.txt", b"some text", "text/plain")},
    )
    assert response.status_code == 400


# --- GET /status ---

def test_status_idle_initially():
    response = client.get("/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "idle"
    assert data["chunks_ingested"] == 0
    assert data["detail"] == ""


@patch("main.get_ingestion_status", return_value={"status": "processing", "chunks_ingested": 0, "detail": ""})
def test_status_returns_processing(mock_status):
    assert client.get("/status").json()["status"] == "processing"


@patch("main.get_ingestion_status", return_value={"status": "ready", "chunks_ingested": 12, "detail": ""})
def test_status_returns_ready_with_chunk_count(mock_status):
    data = client.get("/status").json()
    assert data["status"] == "ready"
    assert data["chunks_ingested"] == 12


@patch("main.get_ingestion_status", return_value={"status": "error", "chunks_ingested": 0, "detail": "Embedding failed"})
def test_status_returns_error_with_detail(mock_status):
    data = client.get("/status").json()
    assert data["status"] == "error"
    assert data["detail"] == "Embedding failed"


# --- POST /answer ---

@patch("main.build_agent")
@patch("main.get_retriever")
@patch("main.is_loaded", return_value=True)
@patch("main.get_ingestion_status", return_value={"status": "ready", "chunks_ingested": 5, "detail": ""})
def test_answer_returns_list(mock_status, mock_loaded, mock_retriever, mock_build_agent):
    mock_agent = MagicMock()
    mock_agent.invoke.return_value = {"answers": MOCK_ANSWERS}
    mock_build_agent.return_value = mock_agent

    response = client.post(
        "/answer",
        files={"questions_file": ("q.json", json.dumps(SAMPLE_QUESTIONS).encode(), "application/json")},
    )
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert data[0]["question"] == SAMPLE_QUESTIONS[0]
    assert "answer" in data[0]


@patch("main.is_loaded", return_value=False)
@patch("main.get_ingestion_status", return_value={"status": "idle", "chunks_ingested": 0, "detail": ""})
def test_answer_without_ingest_returns_400(mock_status, mock_loaded):
    response = client.post(
        "/answer",
        files={"questions_file": ("q.json", json.dumps(SAMPLE_QUESTIONS).encode(), "application/json")},
    )
    assert response.status_code == 400
    assert "ingest" in response.json()["detail"].lower()


@patch("main.get_ingestion_status", return_value={"status": "processing", "chunks_ingested": 0, "detail": ""})
def test_answer_while_processing_returns_409(mock_status):
    response = client.post(
        "/answer",
        files={"questions_file": ("q.json", json.dumps(SAMPLE_QUESTIONS).encode(), "application/json")},
    )
    assert response.status_code == 409
    assert "still being ingested" in response.json()["detail"].lower()


@patch("main.build_agent")
@patch("main.get_retriever")
@patch("main.ingest_text", return_value=5)
def test_answer_with_knowledge_file_ingests_then_answers(mock_ingest, mock_retriever, mock_build_agent):
    mock_agent = MagicMock()
    mock_agent.invoke.return_value = {"answers": MOCK_ANSWERS}
    mock_build_agent.return_value = mock_agent

    response = client.post(
        "/answer",
        files={
            "questions_file": ("q.json", json.dumps(SAMPLE_QUESTIONS).encode(), "application/json"),
            "knowledge_file": ("knowledge.json", SAMPLE_KNOWLEDGE.encode(), "application/json"),
        },
    )
    assert response.status_code == 200
    mock_ingest.assert_called_once()
    mock_build_agent.assert_called_once()


@patch("main.build_agent")
@patch("main.get_retriever")
@patch("main.ingest_text", return_value=3)
def test_answer_with_knowledge_file_does_not_require_prior_ingest(mock_ingest, mock_retriever, mock_build_agent):
    mock_agent = MagicMock()
    mock_agent.invoke.return_value = {"answers": MOCK_ANSWERS}
    mock_build_agent.return_value = mock_agent

    response = client.post(
        "/answer",
        files={
            "questions_file": ("q.json", json.dumps(SAMPLE_QUESTIONS).encode(), "application/json"),
            "knowledge_file": ("knowledge.json", SAMPLE_KNOWLEDGE.encode(), "application/json"),
        },
    )
    assert response.status_code == 200


@patch("main.is_loaded", return_value=True)
@patch("main.get_ingestion_status", return_value={"status": "ready", "chunks_ingested": 5, "detail": ""})
def test_answer_invalid_questions_not_a_list(mock_status, mock_loaded):
    response = client.post(
        "/answer",
        files={"questions_file": ("q.json", b'{"not": "a list"}', "application/json")},
    )
    assert response.status_code == 400
    assert "JSON array of strings" in response.json()["detail"]


@patch("main.is_loaded", return_value=True)
@patch("main.get_ingestion_status", return_value={"status": "ready", "chunks_ingested": 5, "detail": ""})
def test_answer_questions_not_strings(mock_status, mock_loaded):
    response = client.post(
        "/answer",
        files={"questions_file": ("q.json", b'[1, 2, 3]', "application/json")},
    )
    assert response.status_code == 400


@patch("main.build_agent")
@patch("main.get_retriever")
@patch("main.is_loaded", return_value=True)
@patch("main.get_ingestion_status", return_value={"status": "ready", "chunks_ingested": 5, "detail": ""})
def test_answer_agent_invoked_with_correct_state(mock_status, mock_loaded, mock_retriever, mock_build_agent):
    mock_agent = MagicMock()
    mock_agent.invoke.return_value = {"answers": MOCK_ANSWERS}
    mock_build_agent.return_value = mock_agent

    client.post(
        "/answer",
        files={"questions_file": ("q.json", json.dumps(SAMPLE_QUESTIONS).encode(), "application/json")},
    )

    call_args = mock_agent.invoke.call_args[0][0]
    assert call_args["questions"] == SAMPLE_QUESTIONS
    assert call_args["answers"] == []
