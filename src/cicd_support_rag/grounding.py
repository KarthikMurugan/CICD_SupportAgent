"""Structured drafts and deterministic source checks for support answers.

Quotes establish traceability, not semantic entailment. A separate model review
checks claim support; neither check provides a calibrated confidence probability.
"""
from typing import Literal
import re

from langchain_core.documents import Document
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Evidence(StrictModel):
    evidence_id: str = Field(min_length=1)


class Claim(StrictModel):
    text: str = Field(min_length=5, max_length=1200)
    evidence: list[Evidence] = Field(min_length=1, max_length=3)


class Draft(StrictModel):
    decision: Literal["answer", "clarify", "escalate"]
    reason: str = Field(min_length=1, max_length=800)
    claims: list[Claim] = Field(max_length=6)
    clarification: str = Field(max_length=500)

    @model_validator(mode="after")
    def consistent_decision(self):
        if self.decision == "answer" and not self.claims:
            raise ValueError("An answer must contain supported claims")
        if self.decision != "answer" and self.claims:
            raise ValueError("Clarification and escalation must not contain answer claims")
        if self.decision == "clarify" and not self.clarification.strip():
            raise ValueError("A clarification must ask a question")
        return self


class Review(StrictModel):
    fully_answers_question: bool
    every_claim_supported: bool
    required_caveats_preserved: bool
    reason: str = Field(min_length=1, max_length=800)


def evidence_catalog(documents: list[Document]):
    """Bind model-selectable IDs to exact source lines, without model-made quotes."""
    catalog = {}
    for source in documents:
        for index, line in enumerate(source.page_content.splitlines(), 1):
            if line.strip():
                evidence_id = f"{source.metadata['chunk_id']}:line-{index:03d}"
                catalog[evidence_id] = {"document": source, "quote": line.strip()}
    return catalog


def validate_quotes(draft: Draft, documents: list[Document]):
    """Reject missing, fabricated, or unrelated-by-ID quote references."""
    catalog = evidence_catalog(documents)
    for claim in draft.claims:
        for evidence in claim.evidence:
            entry = catalog.get(evidence.evidence_id)
            if entry is None:
                return False, "A citation refers to an excerpt outside the retrieved evidence."
            source = entry["document"]
            if "deprecated" in source.metadata["title"].lower():
                return False, "The answer relies on deprecated documentation."
            if entry["quote"] not in source.page_content:
                return False, "A supporting quote was not found in its cited passage."
    return True, "All supporting quotes match retrieved source passages."


def source_cautions(draft: Draft, documents: list[Document]):
    """Carry restrictions from retrieved parts of the cited baseline passages.

    This conservative lexical rule may include extra cautions from the same
    baseline passage. It does not detect every implicit prerequisite or guarantee that
    the complete source document has been retrieved.
    """
    catalog = evidence_catalog(documents)
    def group(source):
        return source.metadata.get("baseline_chunk_id", source.metadata["chunk_id"])

    cited_groups = {group(catalog[e.evidence_id]["document"])
                    for claim in draft.claims for e in claim.evidence if e.evidence_id in catalog}
    pattern = re.compile(r"\b(?:never|unless|only|restricted|must|required)\b|\bdo\s+not\b", re.I)
    result = []
    seen = set()
    for evidence_id, entry in catalog.items():
        source = entry["document"]
        quote = entry["quote"]
        if group(source) not in cited_groups or not pattern.search(quote):
            continue
        if quote in seen:
            continue
        seen.add(quote)
        result.append({"evidence_id": evidence_id, "document_id": source.metadata["document_id"],
                       "chunk_id": source.metadata["chunk_id"], "quote": quote})
    return result


def render_answer(draft: Draft, documents: list[Document]):
    valid, reason = validate_quotes(draft, documents)
    if draft.decision != "answer" or not valid:
        raise ValueError(reason if not valid else "Only answer drafts can be rendered")
    catalog = evidence_catalog(documents)
    lines = []
    cautions = source_cautions(draft, documents)
    if cautions:
        lines.append("Before you start, note these restrictions from the cited documentation:")
        for caution in cautions:
            lines.append(f"> {caution['quote']} [{caution['document_id']}]\n")
        lines.append("Documented guidance:")
    sources = {}
    for claim in draft.claims:
        ids = []
        for citation in claim.evidence:
            metadata = catalog[citation.evidence_id]["document"].metadata
            document_id = metadata["document_id"]
            if document_id not in ids:
                ids.append(document_id)
            sources[document_id] = metadata["title"]
        lines.append("- " + claim.text + " " + " ".join(f"[{source}]" for source in ids))
    lines.append("\nSources:")
    lines.extend(f"- [{source}] {title}" for source, title in sources.items())
    lines.append("\nLinks, commands, and configurations in this adapted documentation are illustrative.")
    return "\n".join(lines)
