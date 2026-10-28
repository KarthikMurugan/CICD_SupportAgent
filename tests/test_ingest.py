"""Check corpus text normalization without external services or model calls."""
import json
import tempfile
import unittest
from pathlib import Path

from cicd_support_rag.ingest import normalize_corpus


class IngestNormalizationTests(unittest.TestCase):
    def normalize(self, content):
        corpus = {
            "product_name": "PipelineHub",
            "document_count": 1,
            "dataset_version": "test-v1",
            "documents": [{"id": "doc-1", "title": "Deploy", "content": content}],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corpus.json"
            path.write_text(json.dumps(corpus), encoding="utf-8")
            return normalize_corpus(path)

    def test_line_endings_normalize_to_lf(self):
        for content in ("first\nsecond\nthird", "first\r\nsecond\r\nthird",
                        "first\rsecond\rthird", "first\r\nsecond\rthird"):
            with self.subTest(content=repr(content)):
                records, summary = self.normalize(content)
                self.assertEqual(records[0]["page_content"], "first\nsecond\nthird")
                self.assertEqual(summary["body_word_count"], 3)

    def test_blank_lines_and_nonbreaking_space_are_preserved_and_normalized(self):
        records, _ = self.normalize("  first\u00a0line\r\rsecond line  ")
        self.assertEqual(records[0]["page_content"], "first line\n\nsecond line")
        self.assertEqual(records[0]["metadata"]["document_id"], "doc-1")

    def test_normalized_empty_document_remains_excluded(self):
        records, summary = self.normalize(" \r\r\n\u00a0 ")
        self.assertEqual(records, [])
        self.assertEqual(summary["empty_document_ids"], ["doc-1"])
        self.assertEqual(summary["ingested_documents"], 0)


if __name__ == "__main__":
    unittest.main()
