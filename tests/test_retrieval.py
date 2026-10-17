import unittest

from langchain_core.documents import Document
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

from cicd_support_rag.embeddings import split_for_model, token_count
from cicd_support_rag.search import reciprocal_rank_fusion
from cicd_support_rag.workflow import build_workflow


class RetrievalTests(unittest.TestCase):
    def test_model_splits_cover_source_with_correct_offsets(self):
        tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "[CLS]": 1, "[SEP]": 2}, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer.post_processor = TemplateProcessing(
            single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 1), ("[SEP]", 2)],
        )
        text = "Header\n\n```yaml\n" + "key: value\n" * 80 + "```\n\nLast paragraph."
        record = {"page_content": text, "metadata": {
            "document_id": "doc-test", "parent_id": "doc-parent", "title": "Test",
            "chunk_id": "doc-test:chunk-0001", "char_start": 15, "char_end": 15 + len(text),
        }}
        parts = split_for_model(record, tokenizer, max_tokens=30)
        self.assertGreater(len(parts), 1)
        self.assertEqual("".join(p["page_content"] for p in parts), text)
        previous_end = 15
        for part in parts:
            metadata = part["metadata"]
            self.assertLessEqual(token_count(tokenizer, part["page_content"]), 30)
            self.assertEqual(metadata["char_start"], previous_end)
            self.assertEqual(text[metadata["char_start"] - 15:metadata["char_end"] - 15], part["page_content"])
            self.assertEqual(metadata["parent_id"], "doc-parent")
            previous_end = metadata["char_end"]
        self.assertEqual(previous_end, 15 + len(text))

    def test_shared_result_moves_up_and_duplicate_cannot_boost(self):
        combined = reciprocal_rank_fusion([["keyword", "both"], ["semantic", "both"]])
        self.assertEqual(combined[0][0], "both")
        self.assertEqual(reciprocal_rank_fusion([["a", "a"]]), reciprocal_rank_fusion([["a"]]))
        self.assertEqual(reciprocal_rank_fusion([[], []]), [])

    def test_workflow_routes_to_answer_only_when_evidence_sufficient(self):
        docs = [Document(page_content="Supported passage", metadata={"document_id": "doc-test"})]
        answers = []

        def answer(question, documents):
            answers.append(question)
            return "A supported answer [doc-test]"

        supported = build_workflow(lambda question: docs, lambda q, d: (True, "Supported"), answer)
        state = supported.invoke({"question": "Supported question"})
        self.assertEqual(state["status"], "answered")
        unsupported = build_workflow(lambda question: docs, lambda q, d: (False, "Missing evidence"), answer)
        self.assertEqual(unsupported.invoke({"question": "Unknown"})["status"], "needs_human_review")
        empty = build_workflow(lambda question: [], lambda q, d: self.fail("Assessor called without evidence"), answer)
        self.assertEqual(empty.invoke({"question": "Unknown"})["status"], "needs_human_review")
        self.assertEqual(answers, ["Supported question"])


if __name__ == "__main__":
    unittest.main()
