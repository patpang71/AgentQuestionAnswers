import threading
import time

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

from agent import build_agent, AgentState, SYSTEM_PROMPT


def make_state(questions) -> AgentState:
    return {"questions": questions, "answers": []}


def make_retriever(texts: list[str]) -> MagicMock:
    docs = [Document(page_content=t) for t in texts]
    retriever = MagicMock()
    retriever.invoke.return_value = docs
    return retriever


def mock_llm_response(text: str) -> MagicMock:
    response = MagicMock()
    response.content = text
    return response


@patch("agent.ChatOpenAI")
def test_returns_one_answer_per_question(mock_chat):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_llm_response("US Central on GCP.")
    mock_chat.return_value = mock_llm

    agent = build_agent(make_retriever(["Our data centers are in US Central."]))
    result = agent.invoke(make_state(["Where are your data centres?"]))

    assert len(result["answers"]) == 1
    assert result["answers"][0]["question"] == "Where are your data centres?"
    assert result["answers"][0]["answer"] == "US Central on GCP."


@patch("agent.ChatOpenAI")
def test_multiple_questions(mock_chat):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_llm_response("Some answer.")
    mock_chat.return_value = mock_llm

    agent = build_agent(make_retriever(["Context."]))
    result = agent.invoke(make_state(["Question 1", "Question 2"]))

    assert len(result["answers"]) == 2
    # Order of answers matches order of input questions
    assert result["answers"][0]["question"] == "Question 1"
    assert result["answers"][1]["question"] == "Question 2"
    assert all("answer" in a for a in result["answers"])


@patch("agent.ChatOpenAI")
def test_strips_whitespace_from_answer(mock_chat):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_llm_response("  Answer with spaces  ")
    mock_chat.return_value = mock_llm

    agent = build_agent(make_retriever(["Context."]))
    result = agent.invoke(make_state(["Question?"]))

    assert result["answers"][0]["answer"] == "Answer with spaces"


@patch("agent.ChatOpenAI")
def test_empty_questions(mock_chat):
    mock_llm = MagicMock()
    mock_chat.return_value = mock_llm

    agent = build_agent(make_retriever([]))
    result = agent.invoke(make_state([]))

    assert result["answers"] == []
    mock_llm.invoke.assert_not_called()


@patch("agent.ChatOpenAI")
def test_cannot_find_answer_passthrough(mock_chat):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_llm_response(
        "I cannot find the answer from given source"
    )
    mock_chat.return_value = mock_llm

    agent = build_agent(make_retriever(["Unrelated content."]))
    result = agent.invoke(make_state(["Unrelated question?"]))

    assert result["answers"][0]["answer"] == "I cannot find the answer from given source"


@patch("agent.ChatOpenAI")
def test_retriever_called_once_per_question(mock_chat):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_llm_response("Answer.")
    mock_chat.return_value = mock_llm

    retriever = make_retriever(["Context."])
    agent = build_agent(retriever)
    agent.invoke(make_state(["Q1", "Q2", "Q3"]))

    assert retriever.invoke.call_count == 3


@patch("agent.ChatOpenAI")
def test_system_prompt_passed_to_llm(mock_chat):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_llm_response("Answer.")
    mock_chat.return_value = mock_llm

    agent = build_agent(make_retriever(["Context."]))
    agent.invoke(make_state(["Question?"]))

    call_args = mock_llm.invoke.call_args[0][0]
    assert call_args[0].content == SYSTEM_PROMPT


@patch("agent.ChatOpenAI")
def test_retrieved_context_included_in_prompt(mock_chat):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_llm_response("Answer.")
    mock_chat.return_value = mock_llm

    context = "The sky is blue."
    agent = build_agent(make_retriever([context]))
    agent.invoke(make_state(["What color is the sky?"]))

    call_args = mock_llm.invoke.call_args[0][0]
    human_message = call_args[1]
    assert context in human_message.content


def test_build_agent_returns_compiled_graph():
    agent = build_agent(make_retriever([]))
    assert agent is not None


@patch("agent.ChatOpenAI")
def test_llm_configured_with_default_timeout_and_retries(mock_chat):
    build_agent(make_retriever([]))

    _, kwargs = mock_chat.call_args
    assert kwargs["request_timeout"] == 30.0
    assert kwargs["max_retries"] == 2


@patch.dict("os.environ", {"OPENAI_TIMEOUT_SECONDS": "5", "OPENAI_MAX_RETRIES": "0"})
@patch("agent.ChatOpenAI")
def test_llm_timeout_and_retries_configurable_via_env(mock_chat):
    build_agent(make_retriever([]))

    _, kwargs = mock_chat.call_args
    assert kwargs["request_timeout"] == 5.0
    assert kwargs["max_retries"] == 0


@patch("agent.ChatOpenAI")
def test_one_failing_question_does_not_break_others(mock_chat):
    def side_effect(messages):
        human_content = messages[1].content
        if "bad question" in human_content:
            raise RuntimeError("boom")
        return mock_llm_response("Good answer.")

    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = side_effect
    mock_chat.return_value = mock_llm

    agent = build_agent(make_retriever(["Context."]))
    result = agent.invoke(make_state(["good question", "bad question"]))

    answers = {a["question"]: a for a in result["answers"]}
    assert len(answers) == 2
    assert answers["good question"]["answer"] == "Good answer."
    assert answers["good question"]["error"] is False
    assert answers["bad question"]["error"] is True
    assert "error occurred" in answers["bad question"]["answer"]


@patch("agent.ChatOpenAI")
def test_successful_answers_marked_not_error(mock_chat):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_llm_response("Answer.")
    mock_chat.return_value = mock_llm

    agent = build_agent(make_retriever(["Context."]))
    result = agent.invoke(make_state(["Question?"]))

    assert result["answers"][0]["error"] is False


@patch("agent.ChatOpenAI")
def test_concurrent_questions_bounded_by_max_workers(mock_chat, monkeypatch):
    monkeypatch.setattr("agent.MAX_CONCURRENT_QUESTIONS", 3)

    active = 0
    max_active = 0
    lock = threading.Lock()

    def side_effect(messages):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        return mock_llm_response("Answer.")

    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = side_effect
    mock_chat.return_value = mock_llm

    agent = build_agent(make_retriever(["Context."]))
    questions = [f"Question {i}" for i in range(10)]
    result = agent.invoke(make_state(questions))

    assert len(result["answers"]) == 10
    assert max_active <= 3
