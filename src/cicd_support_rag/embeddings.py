"""Local BGE embeddings; no hosted model service is used.

The first use downloads public model/tokenizer weights. Passage text is processed
locally. Pinecone uploads happen separately through the search module.
"""
from pathlib import Path

from fastembed import TextEmbedding
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import LocalEntryNotFoundError
from langchain_core.embeddings import Embeddings
from tokenizers import Tokenizer

MODEL_NAME = "BAAI/bge-small-en-v1.5"
DIMENSION = 384
MAX_INPUT_TOKENS = 512
CHUNK_TOKEN_LIMIT = 400
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def load_tokenizer(cache_dir: Path, allow_download=False):
    options = {"repo_id": MODEL_NAME, "filename": "tokenizer.json",
               "cache_dir": str(cache_dir / "tokenizer")}
    try:
        path = hf_hub_download(**options, local_files_only=True)
    except LocalEntryNotFoundError:
        if not allow_download:
            raise ValueError("Tokenizer is not cached; run cicd_support_rag.prepare first") from None
        path = hf_hub_download(**options)
    tokenizer = Tokenizer.from_file(path)
    # Measure full input; a truncated tokenizer would hide overlong text.
    tokenizer.no_truncation()
    tokenizer.no_padding()
    return tokenizer


def token_count(tokenizer, text):
    return len(tokenizer.encode(text, add_special_tokens=True).ids)


def split_for_model(record, tokenizer, max_tokens=CHUNK_TOKEN_LIMIT):
    """Refine a baseline chunk into exact, nonoverlapping source slices.

    Prefer paragraph, line, sentence, then space boundaries near the midpoint.
    Character splits are a last resort for long unbroken values. Large code
    blocks/procedures may be divided; citations retain their source offsets.
    """
    if max_tokens < 8 or max_tokens > MAX_INPUT_TOKENS:
        raise ValueError("max_tokens must be between 8 and the model input limit")
    text = record["page_content"]

    def spans(start, end):
        content = text[start:end]
        if token_count(tokenizer, content) <= max_tokens:
            yield start, end
            return
        midpoint = len(content) // 2
        cut = None
        for separator in ("\n\n", "\n", ". ", " "):
            candidates = []
            position = content.find(separator)
            while position != -1:
                boundary = position + len(separator)
                if len(content) // 4 <= boundary <= 3 * len(content) // 4:
                    candidates.append(boundary)
                position = content.find(separator, position + len(separator))
            if candidates:
                cut = min(candidates, key=lambda boundary: abs(boundary - midpoint))
                break
        cut = cut or midpoint
        if not 0 < cut < len(content):
            raise ValueError("Cannot split input within token budget")
        yield from spans(start, start + cut)
        yield from spans(start + cut, end)

    parts = list(spans(0, len(text))) if text.strip() else []
    result = []
    for index, (start, end) in enumerate(parts):
        original = record["metadata"]
        content = text[start:end]
        result.append({"page_content": content, "metadata": {
            **original,
            "baseline_chunk_id": original["chunk_id"],
            "chunk_id": f"{original['chunk_id']}:bge-{index + 1:03d}",
            "char_start": original["char_start"] + start,
            "char_end": original["char_start"] + end,
            "word_count": len(content.split()),
            "token_count": token_count(tokenizer, content),
            "oversized": False,
            "split_for_model_limit": len(parts) > 1,
            "embedding_model": MODEL_NAME,
        }})
    if "".join(part["page_content"] for part in result) != text:
        if text.strip():
            raise ValueError("Token refinement changed source text")
    return result


class LocalEmbeddings(Embeddings):
    def __init__(self, cache_dir=Path("data/model-cache"), allow_download=False):
        self.tokenizer = load_tokenizer(Path(cache_dir), allow_download=allow_download)
        self.model = TextEmbedding(
            model_name=MODEL_NAME, cache_dir=str(cache_dir), threads=2,
            local_files_only=not allow_download,
        )

    def embed_documents(self, texts):
        if any(token_count(self.tokenizer, text) > MAX_INPUT_TOKENS for text in texts):
            raise ValueError("Passage exceeds embedding input limit; run preparation first")
        return [vector.tolist() for vector in self.model.embed(texts, batch_size=32)]

    def embed_query(self, text):
        if not text.strip():
            raise ValueError("Question must not be empty")
        prompted = QUERY_PREFIX + text
        if token_count(self.tokenizer, prompted) > MAX_INPUT_TOKENS:
            raise ValueError("Question exceeds the embedding model input limit")
        return next(self.model.embed([prompted])).tolist()
