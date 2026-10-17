"""Original minimal scaffold with injected retrieval, assessment and generation.

The executable support bot now uses build_support_workflow in bot.py, which adds
structured drafts, citation checks, claim review and clarification. This original
graph remains available as a small teaching example. Neither submits tickets.
"""
from collections.abc import Callable
from typing import TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph


class SupportState(TypedDict, total=False):
    question: str
    documents: list[Document]
    evidence_sufficient: bool
    evidence_reason: str
    answer: str
    status: str


Retriever = Callable[[str], list[Document]]
EvidenceAssessor = Callable[[str, list[Document]], tuple[bool, str]]
Answerer = Callable[[str, list[Document]], str]


def build_workflow(
    retrieve: Retriever,
    assess_evidence: EvidenceAssessor,
    answer_with_citations: Answerer,
):
    """Create the graph; callers supply real implementations before use.

    The assessor must evaluate support in retrieved passages. A fusion rank
    or an LLM's self-reported certainty is not a calibrated probability.
    """
    def retrieve_node(state: SupportState):
        return {"documents": retrieve(state["question"])}

    def assess_node(state: SupportState):
        documents = state["documents"]
        if not documents:
            return {"evidence_sufficient": False, "evidence_reason": "No passages retrieved."}
        sufficient, reason = assess_evidence(state["question"], documents)
        return {"evidence_sufficient": sufficient, "evidence_reason": reason}

    def answer_node(state: SupportState):
        return {
            "answer": answer_with_citations(state["question"], state["documents"]),
            "status": "answered",
        }

    def escalate_node(state: SupportState):
        return {
            "answer": "I could not find enough support in the knowledge base. "
                      "Please contact a support engineer and share your question "
                      "and any error message with secrets removed.",
            "status": "needs_human_review",
        }

    graph = StateGraph(SupportState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("assess", assess_node)
    graph.add_node("answer", answer_node)
    graph.add_node("escalate", escalate_node)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "assess")
    graph.add_conditional_edges(
        "assess",
        lambda state: "answer" if state["evidence_sufficient"] else "escalate",
        {"answer": "answer", "escalate": "escalate"},
    )
    graph.add_edge("answer", END)
    graph.add_edge("escalate", END)
    return graph.compile()
