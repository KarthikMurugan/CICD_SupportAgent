"""Validate and normalize the PipelineHub corpus without external services.

Document contents are reference data, never execution instructions.
This stage retains page boundaries; chunking and embedding follow separately.
"""
import argparse
import json
from pathlib import Path


def normalize_corpus(path: Path):
    corpus = json.loads(path.read_text(encoding="utf-8"))
    if corpus.get("product_name") != "PipelineHub":
        raise ValueError("Expected the PipelineHub corpus.")
    documents = corpus["documents"]
    if corpus["document_count"] != len(documents):
        raise ValueError("Declared document count differs from actual count.")
    ids = [doc["id"] for doc in documents]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate document IDs.")
    known_ids = set(ids)
    records = []
    empty_ids = []
    for doc in documents:
        parent_id = doc.get("parent_id")
        if parent_id is not None and parent_id not in known_ids:
            raise ValueError(f"Unknown parent for {doc['id']}: {parent_id}")
        content = (
            doc["content"].replace("\r\n", "\n").replace("\r", "\n")
            .replace("\u00a0", " ").strip()
        )
        if not content:
            empty_ids.append(doc["id"])
            continue
        records.append({
            "page_content": content,
            "metadata": {
                "document_id": doc["id"],
                "parent_id": parent_id,
                "title": doc["title"],
                "platform": "PipelineHub",
                "dataset_version": corpus["dataset_version"],
                "source": f"pipelinehub:{doc['id']}",
                "source_file": path.name,
                "illustrative_links": True,
            },
        })
    summary = {
        "platform": "PipelineHub",
        "input_documents": len(documents),
        "ingested_documents": len(records),
        "empty_document_ids": empty_ids,
        "unavailable_documents": corpus.get("unavailable_documents", []),
        "body_word_count": sum(len(r["page_content"].split()) for r in records),
        "limitations": corpus.get("limitations", []),
    }
    return records, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("project-config.json"))
    parser.add_argument("--output", type=Path, default=Path("data/normalized"))
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    corpus_path = Path(config["corpus_path"])
    if not corpus_path.is_absolute():
        corpus_path = config_path.parent / corpus_path
    records, summary = normalize_corpus(corpus_path)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "documents.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    (args.output / "ingestion-summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

