from typing import TypedDict, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

load_dotenv()

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    questions: List[str]
    answers: List[dict]


SYSTEM_PROMPT = """You are a precise question-answering assistant.
Answer the question ONLY based on the provided context.
If the answer cannot be found in the context, respond with exactly:
I cannot find the answer from given source
Do not infer, assume, or add any information not present in the context."""

MAX_CONCURRENT_QUESTIONS = int(os.getenv("MAX_CONCURRENT_QUESTIONS", "8"))


def build_agent(retriever):
    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
        request_timeout=float(os.getenv("OPENAI_TIMEOUT_SECONDS", "30")),
        max_retries=int(os.getenv("OPENAI_MAX_RETRIES", "2")),
    )

    def answer_one(question: str) -> dict:
        try:
            docs = retriever.invoke(question)
            context = "\n\n".join(doc.page_content for doc in docs)
            user_prompt = f"""Context:
{context}

Question: {question}

Answer based solely on the context above. Be concise."""
            response = llm.invoke(
                [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=user_prompt),
                ]
            )
            return {"question": question, "answer": response.content.strip(), "error": False}
        except Exception:
            logger.exception("Failed to answer question: %r", question)
            return {
                "question": question,
                "answer": "An error occurred while answering this question. Please try again.",
                "error": True,
            }

    def retrieve_and_answer(state: AgentState) -> AgentState:
        questions = state["questions"]
        answers = [None] * len(questions)

        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_QUESTIONS) as executor:
            futures = {executor.submit(answer_one, q): i for i, q in enumerate(questions)}
            for future in as_completed(futures):
                answers[futures[future]] = future.result()

        return {"answers": answers}

    graph = StateGraph(AgentState)
    graph.add_node("retrieve_and_answer", retrieve_and_answer)
    graph.set_entry_point("retrieve_and_answer")
    graph.add_edge("retrieve_and_answer", END)
    return graph.compile()
