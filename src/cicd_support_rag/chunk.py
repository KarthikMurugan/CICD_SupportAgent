"""Prepare citation-ready passages without a model or external dependencies.

Paragraphs and fenced code blocks stay intact. The word target is a soft limit:
an oversized block is retained and flagged for review, never silently truncated.
Offsets refer to normalized document text, not the original JSON file.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

from cicd_support_rag.ingest import normalize_corpus


def block_spans(text):
    """Yield paragraph spans, ignoring blank lines inside fenced code."""
    start = None
    position = 0
    fence = None
    for line in text.splitlines(keepends=True):
        marker = re.match(r"^\s*(`{3,}|~{3,})", line)
        if marker:
            run = marker.group(1)
            if fence is None:
                fence = run
            elif run[0] == fence[0] and len(run) >= len(fence):
                fence = None
        if line.strip() or fence is not None:
            if start is None:
                start = position
        elif start is not None:
            yield start, position
            start = None
        position += len(line)
    if start is not None:
        yield start, len(text)


def chunk_document(record, target_words=450):
    """Pack complete paragraphs; carry document metadata into every passage.

    No overlap in this baseline, so corpus coverage and duplication are auditable.
    Splitting uses words rather than model tokens; tune after choosing embeddings.
    """
    if target_words < 1:
        raise ValueError("target_words must be positive")
    text = record["page_content"]
    spans = list(block_spans(text))
    groups = []
    group_start = group_end = None
    word_count = 0
    for start, end in spans:
        count = len(text[start:end].split())
        if group_start is not None and word_count + count > target_words:
            groups.append((group_start, group_end))
            group_start = None
            word_count = 0
        if group_start is None:
            group_start = start
        group_end = end
        word_count += count
    if group_start is not None:
        groups.append((group_start, group_end))

    chunks = []
    for index, (start, end) in enumerate(groups):
        content = text[start:end].rstrip()
        end = start + len(content)
        words = len(content.split())
        chunks.append({
            "page_content": content,
            "metadata": {
                **record["metadata"],
                "chunk_id": f"{record['metadata']['document_id']}:chunk-{index + 1:04d}",
                "chunk_index": index,
                "char_start": start,
                "char_end": end,
                "word_count": words,
                "oversized": words > target_words,
            },
        })
    return chunks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("project-config.json"))
    parser.add_argument("--output", type=Path, default=Path("data/chunked"))
    parser.add_argument("--target-words", type=int, default=450)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    corpus_path = Path(config["corpus_path"])
    if not corpus_path.is_absolute():
        corpus_path = config_path.parent / corpus_path
    records, ingestion_summary = normalize_corpus(corpus_path)
    chunks = [chunk for record in records
              for chunk in chunk_document(record, args.target_words)]
    # Verify exact source slices and full word coverage before writing output.
    originals = {r["metadata"]["document_id"]: r["page_content"] for r in records}
    for chunk in chunks:
        metadata = chunk["metadata"]
        original = originals[metadata["document_id"]]
        if original[metadata["char_start"]:metadata["char_end"]] != chunk["page_content"]:
            raise ValueError("Chunk does not match its source span")
    for document_id, original in originals.items():
        reconstructed = " ".join(c["page_content"] for c in chunks
                                 if c["metadata"]["document_id"] == document_id)
        if reconstructed.split() != original.split():
            raise ValueError(f"Incomplete coverage for {document_id}")
    oversized = [c["metadata"]["chunk_id"] for c in chunks if c["metadata"]["oversized"]]
    summary = {
        "ingestion": ingestion_summary,
        "corpus_sha256": hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
        "strategy": "paragraph_and_fenced_code_blocks_v1",
        "target_words": args.target_words,
        "overlap_words": 0,
        "chunk_count": len(chunks),
        "documents_with_chunks": len({c["metadata"]["document_id"] for c in chunks}),
        "chunk_word_count": sum(c["metadata"]["word_count"] for c in chunks),
        "max_chunk_words": max((c["metadata"]["word_count"] for c in chunks), default=0),
        "oversized_chunk_ids": oversized,
        "source_spans_verified": True,
        "word_coverage_verified": True,
        "limitations": [
            "Word counts are not model token counts.",
            "Long paragraphs and procedures may require further segmentation before embedding.",
            "Paragraph boundaries do not guarantee complete procedure context.",
            "Retrieval quality and answer quality have not been evaluated.",
        ],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "chunks.jsonl").open("w", encoding="utf-8") as stream:
        for chunk in chunks:
            stream.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    (args.output / "chunk-summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "ingestion"}, indent=2))


if __name__ == "__main__":
    main()
