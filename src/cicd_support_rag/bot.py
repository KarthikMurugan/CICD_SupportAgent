"""Support bot: Pinecone hybrid retrieval, configured model answers, source checks.

Run from the project directory:
  python -m cicd_support_rag.bot "How do I register an OAuth client?"
  python -m cicd_support_rag.bot "..." --json

Human escalation prepares a review packet; it does not submit a real ticket.
Evidence rules and model review are provisional until evaluated on labeled data.
"""
import argparse
import json
from pathlib import Path
import re
import time
from typing import TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph

from cicd_support_rag.generation import (
    DRAFT_SYSTEM, PROMPT_VERSION, REVIEW_SYSTEM, GenerationError, generation_client,
)
from cicd_support_rag.grounding import Draft, Review, evidence_catalog, render_answer, source_cautions, validate_quotes
from cicd_support_rag.search import SupportRetriever


class BotState(TypedDict, total=False):
    question: str
    documents: list[Document]
    draft: dict
    decision: str
    reason: str
    answer: str
    status: str
    handoff: dict


def passage_payload(documents):
    return [{"chunk_id": d.metadata["chunk_id"], "document_id": d.metadata["document_id"],
             "parent_id": d.metadata.get("parent_id"), "title": d.metadata["title"],
             "char_start": d.metadata.get("char_start"), "char_end": d.metadata.get("char_end"),
             "text": d.page_content} for d in documents]


def model_passages(documents):
    catalog = evidence_catalog(documents)
    return [{"chunk_id": d.metadata["chunk_id"], "document_id": d.metadata["document_id"],
             "title": d.metadata["title"], "excerpts": [
                 {"evidence_id": evidence_id, "text": entry["quote"]}
                 for evidence_id, entry in catalog.items() if entry["document"] is d
             ]} for d in documents]


def vague_failure_report(question):
    """A provisional intent rule for brief failures without an identified step."""
    failure = re.search(r"\b(?:failed|failure|broken)\b|\bnot working\b|\bdoesn't work\b", question, re.I)
    operation = re.search(r"\b(?:build|deploy(?:ment)?|preflight|test(?:ing)?|oauth|login|authentication|registration|onboarding|artifact|rollback|release)\b", question, re.I)
    error_code = re.search(r"\b[A-Z][A-Z_]+[-_]?\d+\b", question)
    return bool(failure and not operation and not error_code and len(question.split()) <= 12)


def build_support_workflow(retrieve, model):
    def retrieve_node(state):
        # Known-deprecated pages cannot establish a current supported procedure.
        documents = [d for d in retrieve(state["question"])
                     if "deprecated" not in d.metadata["title"].lower()][:6]
        return {"documents": documents}

    def draft_node(state):
        if vague_failure_report(state["question"]):
            draft = Draft(decision="clarify", reason="The failed operation and error details are missing.",
                          claims=[], clarification="Which step failed (build, deployment, test, or login), and what error message do you see?")
            return {"draft": draft.model_dump(), "decision": "clarify", "reason": draft.reason}
        if not state["documents"]:
            return {"decision": "escalate", "reason": "No eligible evidence was retrieved."}
        try:
            draft = model.complete(Draft, DRAFT_SYSTEM, {
                "question": state["question"], "passages": model_passages(state["documents"]),
            })
            return {"draft": draft.model_dump(), "decision": draft.decision, "reason": draft.reason}
        except GenerationError as error:
            return {"decision": "escalate", "reason": str(error)}

    def verify_node(state):
        draft = Draft.model_validate(state["draft"])
        valid, reason = validate_quotes(draft, state["documents"])
        if not valid:
            return {"decision": "escalate", "reason": reason}
        try:
            review = model.complete(Review, REVIEW_SYSTEM, {
                "question": state["question"], "draft": state["draft"],
                "passages": model_passages(state["documents"]),
            })
        except GenerationError as error:
            return {"decision": "escalate", "reason": str(error)}
        if not review.fully_answers_question or not review.every_claim_supported or not review.required_caveats_preserved:
            return {"decision": "escalate", "reason": review.reason}
        return {"decision": "answer", "reason": reason + " " + review.reason}

    def answer_node(state):
        draft = Draft.model_validate(state["draft"])
        return {"answer": render_answer(draft, state["documents"]), "status": "answered"}

    def clarify_node(state):
        return {"answer": state["draft"]["clarification"], "status": "needs_clarification"}

    def escalate_node(state):
        return {
            "status": "needs_human_review",
            "answer": "I couldn't find enough verified evidence to answer this reliably. "
                      "Please share this question with a PipelineHub support engineer. "
                      "No support ticket has been submitted.",
            "handoff": {"question": state["question"], "reason": state["reason"],
                        "source_ids": list(dict.fromkeys(d.metadata["document_id"] for d in state["documents"])),
                        "submitted": False},
        }

    graph = StateGraph(BotState)
    for name, node in (("retrieve", retrieve_node), ("draft", draft_node), ("verify", verify_node),
                       ("answer", answer_node), ("clarify", clarify_node), ("escalate", escalate_node)):
        graph.add_node(name, node)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "draft")
    graph.add_conditional_edges("draft", lambda s: s["decision"], {
        "answer": "verify", "clarify": "clarify", "escalate": "escalate",
    })
    graph.add_conditional_edges("verify", lambda s: s["decision"], {"answer": "answer", "escalate": "escalate"})
    for name in ("answer", "clarify", "escalate"):
        graph.add_edge(name, END)
    return graph.compile()


def ask(question, backend="pinecone", model=None, on_stage=None, retrieval_mode="hybrid"):
    if not question.strip():
        raise ValueError("Enter a support question")
    started = time.monotonic()
    retriever = SupportRetriever(backend=backend)
    generator = model or generation_client()
    workflow = build_support_workflow(
        lambda q: retriever.retrieve(q, mode=retrieval_mode, limit=10), generator,
    )
    state = {"question": question}
    for update in workflow.stream(state, stream_mode="updates"):
        for node, values in update.items():
            state.update(values)
            if on_stage:
                on_stage(node)
    draft = state.get("draft")
    catalog = evidence_catalog(state["documents"])
    supporting_quotes = []
    if state["status"] == "answered":
        for claim in draft["claims"]:
            for evidence in claim["evidence"]:
                entry = catalog[evidence["evidence_id"]]
                supporting_quotes.append({"claim": claim["text"], "evidence_id": evidence["evidence_id"],
                                          "quote": entry["quote"],
                                          "document_id": entry["document"].metadata["document_id"],
                                          "parent_id": entry["document"].metadata.get("parent_id"),
                                          "title": entry["document"].metadata["title"],
                                          "chunk_id": entry["document"].metadata["chunk_id"]})
    return {
        "question": question, "status": state["status"], "answer": state["answer"],
        "evidence_reason": state["reason"], "draft": draft,
        "supporting_quotes": supporting_quotes,
        "source_cautions": source_cautions(Draft.model_validate(draft), state["documents"])
                           if state["status"] == "answered" else [],
        "answer_rendering_version": "source-cautions-v1",
        "retrieved_sources": passage_payload(state["documents"]),
        "handoff": state.get("handoff"), "prompt_version": PROMPT_VERSION,
        "generation_model": getattr(generator, "model", "injected_test_model"),
        "retrieval_mode": retrieval_mode, "vector_backend": backend, "elapsed_seconds": round(time.monotonic() - started, 3),
        "confidence_calibrated": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--backend", choices=["pinecone", "local-check"], default="pinecone")
    parser.add_argument("--mode", choices=["keyword", "semantic", "hybrid"], default="hybrid")
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--save", type=Path, help="Save answer and evidence to a JSON report")
    args = parser.parse_args()
    try:
        result = ask(args.question, backend=args.backend, retrieval_mode=args.mode)
    except (ValueError, FileNotFoundError) as error:
        parser.exit(2, f"Setup needed: {error}\n")
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    if args.as_json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(result["answer"])
        print(f"\nStatus: {result['status']} | Elapsed: {result['elapsed_seconds']} seconds")


if __name__ == "__main__":
    main()
