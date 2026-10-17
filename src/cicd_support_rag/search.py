"""BM25 + Pinecone retrieval with RRF; returns source passages, not answers.

Prepare local embeddings first. Commands:
  python -m cicd_support_rag.search upload
  python -m cicd_support_rag.search query "How do I register an OAuth client?"
  python -m cicd_support_rag.search query "..." --backend local-check

The local-check backend uses cached vectors to verify the pipeline before a
Pinecone connection is configured; it is not a Pinecone integration test.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import re

from dotenv import load_dotenv
from langchain_core.documents import Document
import numpy as np
from rank_bm25 import BM25Plus

from cicd_support_rag.embeddings import DIMENSION, MODEL_NAME, LocalEmbeddings


def tokens(text):
    # Retain whole error/config identifiers as well as their word components.
    return re.findall(r"[a-z0-9]+(?:[_.:/-][a-z0-9]+)+|[a-z0-9]+", text.lower())


def reciprocal_rank_fusion(rankings, limit=5, rank_constant=60):
    """Combine ranks. Scores are ordering signals, never probabilities."""
    if limit < 1 or rank_constant < 1:
        raise ValueError("limit and rank_constant must be positive")
    scores = defaultdict(float)
    for ranking in rankings:
        seen = set()
        for rank, chunk_id in enumerate(ranking, 1):
            if chunk_id not in seen:
                scores[chunk_id] += 1 / (rank_constant + rank)
                seen.add(chunk_id)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]


class SupportRetriever:
    def __init__(self, directory=Path("data/embedded"), backend="pinecone"):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        if self.manifest["embedding_model"] != MODEL_NAME or self.manifest["dimension"] != DIMENSION:
            raise ValueError("Prepared embeddings use a different model; rebuild before searching")
        for name, key in (("chunks.jsonl", "chunks_sha256"), ("vectors.npy", "vectors_sha256")):
            if hashlib.sha256((self.directory / name).read_bytes()).hexdigest() != self.manifest[key]:
                raise ValueError(f"{name} differs from its manifest; rebuild prepared data")
        self.chunks = [json.loads(line) for line in (self.directory / "chunks.jsonl").read_text().splitlines()]
        self.by_id = {c["metadata"]["chunk_id"]: c for c in self.chunks}
        if len(self.by_id) != len(self.chunks):
            raise ValueError("Duplicate chunk IDs")
        self.vectors = np.load(self.directory / "vectors.npy", allow_pickle=False)
        if self.vectors.shape != (len(self.chunks), DIMENSION):
            raise ValueError("Vector rows do not match chunk records")
        self.keyword = BM25Plus([
            tokens(c["metadata"]["title"] + " " + c["page_content"]) for c in self.chunks
        ], delta=0)
        if backend not in {"pinecone", "local-check"}:
            raise ValueError("Unknown backend")
        self.backend = backend
        self.embedder = None
        self.index = None

    def connect(self):
        """Connect only to a user-configured, compatible existing dense index."""
        from pinecone import Pinecone

        load_dotenv(Path(__file__).resolve().parents[2] / ".env")
        key = os.environ.get("PINECONE_API_KEY")
        name = os.environ.get("PINECONE_INDEX_NAME")
        if not key or not name:
            raise ValueError("Set PINECONE_API_KEY and PINECONE_INDEX_NAME in the project's .env file")
        self.client = Pinecone(api_key=key)
        description = self.client.describe_index(name)
        if description.dimension != DIMENSION or description.metric != "cosine":
            raise ValueError("Use a dense Pinecone index with 384 dimensions and cosine metric")
        self.index = self.client.Index(host=description.host)

    @property
    def namespace(self):
        return self.manifest["pinecone_namespace"]

    def upload(self):
        if self.index is None:
            self.connect()
        for start in range(0, len(self.chunks), 100):
            batch = []
            for offset, chunk in enumerate(self.chunks[start:start + 100], start):
                metadata = {k: v for k, v in chunk["metadata"].items() if v is not None}
                metadata.update({
                    "text": chunk["page_content"],
                    "dataset_fingerprint": self.manifest["chunks_sha256"],
                })
                batch.append({"id": metadata["chunk_id"], "values": self.vectors[offset].tolist(),
                              "metadata": metadata})
            self.index.upsert(vectors=batch, namespace=self.namespace)
        return {"submitted_vectors": len(self.chunks), "namespace": self.namespace,
                "note": "Pinecone may take time to make upserts visible to queries."}

    def keyword_ranking(self, question, limit):
        scores = self.keyword.get_scores(tokens(question))
        return [self.chunks[i]["metadata"]["chunk_id"]
                for i in np.argsort(-scores, kind="stable")[:limit] if scores[i] > 0]

    def dense_ranking(self, question, limit):
        if self.backend == "pinecone" and self.index is None:
            self.connect()
        if self.embedder is None:
            self.embedder = LocalEmbeddings()
        query = self.embedder.embed_query(question)
        if self.backend == "local-check":
            scores = self.vectors @ np.asarray(query, dtype="float32")
            return [self.chunks[i]["metadata"]["chunk_id"] for i in np.argsort(-scores)[:limit]]
        response = self.index.query(
            vector=query, top_k=limit, namespace=self.namespace,
            filter={"dataset_fingerprint": {"$eq": self.manifest["chunks_sha256"]}},
            include_metadata=False,
        )
        ranking = [match.id for match in response.matches]
        if any(chunk_id not in self.by_id for chunk_id in ranking):
            raise ValueError("Pinecone results do not match the local corpus")
        return ranking

    def retrieve(self, question, mode="hybrid", limit=5, candidates=15):
        if not question.strip():
            raise ValueError("Question must not be empty")
        if limit < 1 or candidates < limit:
            raise ValueError("Require candidates >= limit >= 1")
        if mode == "keyword":
            rankings = [self.keyword_ranking(question, candidates)]
        elif mode == "semantic":
            rankings = [self.dense_ranking(question, candidates)]
        elif mode == "hybrid":
            rankings = [self.keyword_ranking(question, candidates), self.dense_ranking(question, candidates)]
        else:
            raise ValueError("Unknown retrieval mode")
        result = []
        for chunk_id, score in reciprocal_rank_fusion(rankings, limit=limit):
            chunk = self.by_id[chunk_id]
            result.append(Document(page_content=chunk["page_content"], metadata={
                **chunk["metadata"], "rank_fusion_score": score,
                "retrieval_mode": mode, "vector_backend": self.backend,
            }))
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("upload", help="Upload prepared passages and vectors to your configured Pinecone index")
    query = sub.add_parser("query", help="Show retrieved passages with citation metadata")
    query.add_argument("question")
    query.add_argument("--backend", choices=["pinecone", "local-check"], default="pinecone")
    query.add_argument("--mode", choices=["keyword", "semantic", "hybrid"], default="hybrid")
    query.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()
    try:
        retriever = SupportRetriever(backend=getattr(args, "backend", "pinecone"))
        if args.command == "upload":
            print(json.dumps(retriever.upload(), indent=2))
        else:
            documents = retriever.retrieve(args.question, mode=args.mode, limit=args.limit)
            print(json.dumps([{"text": d.page_content, "metadata": d.metadata} for d in documents], indent=2))
    except (ValueError, FileNotFoundError) as error:
        parser.exit(2, f"Setup needed: {error}\n")


if __name__ == "__main__":
    main()
