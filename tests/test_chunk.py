"""Check citation traceability, coverage, and block preservation."""
import unittest

from cicd_support_rag.chunk import chunk_document


class ChunkTests(unittest.TestCase):
    def record(self, text):
        return {"page_content": text, "metadata": {
            "document_id": "doc-test", "parent_id": "doc-parent", "title": "Deploy",
        }}

    def test_source_coverage_and_citations(self):
        text = "First paragraph has words.\n\nSecond paragraph.\n\nFinal paragraph."
        chunks = chunk_document(self.record(text), target_words=3)
        self.assertEqual(len(chunks), 3)
        self.assertEqual(" ".join(c["page_content"] for c in chunks).split(), text.split())
        for chunk in chunks:
            meta = chunk["metadata"]
            self.assertEqual(text[meta["char_start"]:meta["char_end"]], chunk["page_content"])
            self.assertEqual(meta["parent_id"], "doc-parent")
            self.assertEqual(meta["title"], "Deploy")
        self.assertEqual(len({c["metadata"]["chunk_id"] for c in chunks}), 3)

    def test_fenced_code_with_blank_line_stays_intact(self):
        code = "```yaml\nsteps:\n\n  - deploy: staging\n```"
        chunks = chunk_document(self.record("Intro.\n\n" + code + "\n\nEnd."), 2)
        self.assertEqual(chunks[1]["page_content"], code)
        self.assertTrue(chunks[1]["metadata"]["oversized"])

    def test_empty_input_and_invalid_target(self):
        self.assertEqual(chunk_document(self.record(" \n\n")), [])
        with self.assertRaises(ValueError):
            chunk_document(self.record("Hello"), 0)


if __name__ == "__main__":
    unittest.main()
