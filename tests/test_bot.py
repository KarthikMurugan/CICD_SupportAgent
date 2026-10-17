from collections import deque
import unittest

from langchain_core.documents import Document
from pydantic import ValidationError

from cicd_support_rag.bot import build_support_workflow
from cicd_support_rag.generation import GenerationError, OllamaClient
from cicd_support_rag.grounding import Claim, Draft, Evidence, Review


class FixtureModel:
    def __init__(self, *responses):
        self.responses = deque(responses)
        self.calls = 0

    def complete(self, schema, system, payload):
        self.calls += 1
        result = self.responses.popleft()
        if isinstance(result, Exception):
            raise result
        return result


class BotTests(unittest.TestCase):
    def source(self):
        return Document(
            page_content="PipelineHub projects can be paused from the project dashboard.",
            metadata={"document_id": "doc-0001", "chunk_id": "doc-0001:chunk-0001", "title": "Project Guide"},
        )

    def draft(self, evidence_id="doc-0001:chunk-0001:line-001", text=None):
        return Draft(decision="answer", reason="The procedure is described in the evidence.",
                     clarification="", claims=[Claim(
                         text=text or "Pause the project from its project dashboard.",
                         evidence=[Evidence(evidence_id=evidence_id)],
                     )])

    def run_graph(self, model, documents=None):
        graph = build_support_workflow(lambda q: documents if documents is not None else [self.source()], model)
        return graph.invoke({"question": "How can I pause my project?"})

    def test_verified_answer_has_real_document_id_and_title(self):
        model = FixtureModel(self.draft(), Review(fully_answers_question=True, every_claim_supported=True, required_caveats_preserved=True, reason="Supported."))
        result = self.run_graph(model)
        self.assertEqual(result["status"], "answered")
        self.assertIn("[doc-0001] Project Guide", result["answer"])
        self.assertEqual(model.calls, 2)

    def test_fabricated_quote_and_unknown_citation_are_rejected(self):
        with self.assertRaises(ValidationError):
            Evidence.model_validate({"evidence_id": "doc-0001:chunk-0001:line-001", "quote": "A fabricated quotation"})
        model = FixtureModel(self.draft(evidence_id="invented"))
        result = self.run_graph(model)
        self.assertEqual(result["status"], "needs_human_review")
        self.assertEqual(model.calls, 1)
        self.assertFalse(result["handoff"]["submitted"])

    def test_real_quote_does_not_bypass_claim_entailment_review(self):
        model = FixtureModel(self.draft(text="All paused projects receive a refund."),
                             Review(fully_answers_question=False, every_claim_supported=False, required_caveats_preserved=True, reason="Refunds are not supported."))
        result = self.run_graph(model)
        self.assertEqual(result["status"], "needs_human_review")
        self.assertNotIn("receive a refund", result["answer"])

    def test_empty_and_deprecated_evidence_do_not_call_model(self):
        deprecated = self.source()
        deprecated.metadata["title"] = "Project Guide - Deprecated"
        for sources in ([], [deprecated]):
            model = FixtureModel()
            self.assertEqual(self.run_graph(model, sources)["status"], "needs_human_review")
            self.assertEqual(model.calls, 0)

    def test_omitted_required_caveat_blocks_an_otherwise_supported_answer(self):
        model = FixtureModel(self.draft(), Review(fully_answers_question=True, every_claim_supported=True,
                             required_caveats_preserved=False, reason="Required prerequisite is missing."))
        result = self.run_graph(model)
        self.assertEqual(result["status"], "needs_human_review")

    def test_source_restriction_survives_a_model_review_that_misses_it(self):
        source = self.source()
        restriction = "Never request STAGE unless you are an authorized platform developer."
        source.page_content += "\n" + restriction
        model = FixtureModel(self.draft(), Review(fully_answers_question=True, every_claim_supported=True,
                             required_caveats_preserved=True, reason="Model review overlooked the restriction."))
        result = self.run_graph(model, [source])
        self.assertEqual(result["status"], "answered")
        self.assertIn(restriction, result["answer"])

    def test_ambiguous_query_returns_clarification(self):
        model = FixtureModel(Draft(decision="clarify", reason="Project type is missing.", claims=[],
                                   clarification="Which project type are you using?"))
        result = self.run_graph(model)
        self.assertEqual(result["status"], "needs_clarification")
        self.assertEqual(result["answer"], "Which project type are you using?")

    def test_brief_vague_failure_asks_details_without_a_model_call(self):
        model = FixtureModel()
        graph = build_support_workflow(lambda q: [self.source()], model)
        result = graph.invoke({"question": "My pipeline failed. What now?"})
        self.assertEqual(result["status"], "needs_clarification")
        self.assertEqual(model.calls, 0)

    def test_invalid_or_unavailable_model_fails_closed(self):
        for responses in ((GenerationError("Invalid response"),),
                          (self.draft(), GenerationError("Local model unavailable"))):
            result = self.run_graph(FixtureModel(*responses))
            self.assertEqual(result["status"], "needs_human_review")

    def test_escalation_does_not_render_partial_answer(self):
        model = FixtureModel(Draft(decision="escalate", reason="Requested evidence is missing.", claims=[], clarification=""))
        self.assertEqual(self.run_graph(model)["status"], "needs_human_review")

    def test_remote_model_endpoint_is_not_accepted_for_local_provider(self):
        with self.assertRaises(ValueError):
            OllamaClient(base_url="https://remote.example.com")


if __name__ == "__main__":
    unittest.main()
