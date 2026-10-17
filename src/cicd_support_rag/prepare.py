"""Refine baseline chunks to the model budget, then generate local embeddings.

Run: python -m cicd_support_rag.prepare
Does not connect to Pinecone or generate support answers.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from cicd_support_rag.embeddings import (
    CHUNK_TOKEN_LIMIT, DIMENSION, MODEL_NAME, LocalEmbeddings, split_for_model,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/chunked/chunks.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/embedded"))
    parser.add_argument("--cache", type=Path, default=Path("data/model-cache"))
    args = parser.parse_args()
    start = time.monotonic()
    baseline = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    embedder = LocalEmbeddings(args.cache, allow_download=True)
    chunks = [part for chunk in baseline
              for part in split_for_model(chunk, embedder.tokenizer)]
    print(f"Prepared {len(chunks)} token-bounded passages; embedding locally...", flush=True)
    vectors = np.asarray(embedder.embed_documents([c["page_content"] for c in chunks]), dtype="float32")
    if vectors.shape != (len(chunks), DIMENSION) or not np.isfinite(vectors).all():
        raise ValueError("Embedding dimensions or values are invalid")
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=0.01):
        raise ValueError("Expected normalized BGE embeddings")
    args.output.mkdir(parents=True, exist_ok=True)
    corpus_hash = hashlib.sha256(args.input.read_bytes()).hexdigest()
    manifest = {
        "embedding_provider": "local_fastembed",
        "embedding_model": MODEL_NAME,
        "dimension": DIMENSION,
        "metric": "cosine",
        "baseline_chunk_count": len(baseline),
        "chunk_count": len(chunks),
        "documents": len({c["metadata"]["document_id"] for c in chunks}),
        "max_chunk_tokens": max(c["metadata"]["token_count"] for c in chunks),
        "chunk_token_limit": CHUNK_TOKEN_LIMIT,
        "overlap_tokens": 0,
        "baseline_chunks_sha256": corpus_hash,
        "pinecone_namespace": f"pipelinehub-bge-small-v1-{corpus_hash[:12]}",
        "elapsed_seconds": round(time.monotonic() - start, 2),
        "limitations": [
            "Large code blocks and procedures may span multiple passages.",
            "Retrieval relevance has not been evaluated on labeled queries.",
        ],
    }
    chunks_text = "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in chunks)
    manifest["chunks_sha256"] = hashlib.sha256(chunks_text.encode()).hexdigest()
    (args.output / "chunks.jsonl").write_text(chunks_text, encoding="utf-8")
    np.save(args.output / "vectors.npy", vectors, allow_pickle=False)
    manifest["vectors_sha256"] = hashlib.sha256((args.output / "vectors.npy").read_bytes()).hexdigest()
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
