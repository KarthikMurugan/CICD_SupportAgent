# PipelineHub Customer Support Hybrid RAG

A local support chatbot for the adapted PipelineHub CI/CD documentation. It combines BM25 keyword search and Pinecone semantic search, generates source-cited answers with Nebius, and asks for clarification or prepares a human-review message when evidence is insufficient.

Built for The Gen Academy Week 2 Project 4. The platform documentation is illustrative: its example.com links, commands, and configurations are not validated operational instructions.

## Architecture

1. Load the canonical JSON corpus and exclude empty document bodies.
2. Create paragraph/code-aware chunks, then refine them to the embedding tokenizer limit.
3. Generate embeddings locally with `BAAI/bge-small-en-v1.5` (384 dimensions).
4. Retrieve with BM25Plus and Pinecone, combining ranks with reciprocal rank fusion.
5. Use a LangGraph workflow to draft, check evidence, and answer, clarify, or escalate.
6. Display responses, sources, and review packets in a local browser UI.

Generation defaults to Nebius `Qwen/Qwen3.5-397B-A17B`. Local Ollama `qwen3:8b` remains an explicit alternative. Fusion scores and model review are not calibrated confidence probabilities. Human review creates a message and packet; it does not submit a ticket.

## Setup

Use Python 3.11 or newer. Run commands from the repository root.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[retrieval]'
cp .env.example .env
```

Fill in the Pinecone and Nebius credentials in `.env`. Create or select a Pinecone dense index with **384 dimensions and cosine similarity**. The example index name is `pipelinehub-support`.

For the optional local provider, install Ollama separately, pull `qwen3:8b`, and set `GENERATION_PROVIDER=ollama`. Nebius credentials are not required for that provider.

## Prepare the knowledge base

Obtain the course's adapted `PipelineHub-corpus.json`. The full corpus is not bundled in this repository. Do not mix it with TXT/HTML copies or ingest evaluation answers.

The default `project-config.json` expects the canonical file at `../sources/PipelineHub-corpus.json`. Set `corpus_path` to its actual location if using another layout. Keep synced course source files read-only.

```bash
python -m cicd_support_rag.ingest
python -m cicd_support_rag.chunk
python -m cicd_support_rag.prepare
```

The first embedding preparation downloads model weights. These steps produced **131 usable documents and 1,554 final passages** for the recorded corpus. Local outputs and weights are ignored by Git.

To upload those passages, vectors, and citation metadata to your configured Pinecone index:

```bash
python -m cicd_support_rag.search upload
```

This command writes to your Pinecone account. Preparation alone does not upload data. The namespace is derived from the prepared corpus fingerprint.

## Run the chatbot

```bash
python -m cicd_support_rag.ui_server
```

Open <http://127.0.0.1:8765/>. On macOS, `Open PipelineHub Chat.command` provides a launcher after setup. The server binds to localhost; browser history is stored in that browser.

For command-line use:

```bash
python -m cicd_support_rag.bot 'What format is accepted for a free form tag?' --mode hybrid
```

When Nebius is selected, the question and retrieved documentation excerpts are sent to Nebius. The Pinecone namespace holds document passages and vectors. API keys remain server-side.

## Tests

```bash
python -m unittest discover -s tests -v
```

The recorded project environment passed **27 tests**, covering chunking, retrieval fusion, grounding/routing, provider error handling, and the local HTTP interface.

## Evaluation

The frozen golden dataset is under `evaluation/golden-v1/`. It contains 20 newly authored corpus-grounded questions: 6 exact-match, 6 semantic troubleshooting, 4 multi-passage, 2 clarification, and 2 unsupported escalation cases. Source spans and dataset hashes were verified; independent human/SME validation is pending.

The initial run is under `evaluation/runs/2026-10-17-run01/`. It saves all **60 responses** for keyword, semantic, and hybrid retrieval, timings, tokens, sources, criterion-level judgments, and an Excel review workbook.

| Metric | Keyword | Semantic | Hybrid |
| --- | ---: | ---: | ---: |
| Correct answer/clarify/escalate action | 75% | 85% | 95% |
| Median end-to-end response time | 6.9 s | 7.4 s | 7.5 s |
| Strictly resolved answerable cases | 6/16 | 6/16 | 5/16 |
| Simulated FCR across all 20 queries | 30% | 30% | 25% |

Quality scores are provisional primary-assistant AI judgments, not independent human validation or real customer FCR. The frozen compound rubric requires every detail, including Edit/Save steps in some cases, so useful partial answers can fail strict resolution. One incomplete semantic generation was retained rather than retried. Latency is a single trial per question/mode with two requests in flight.

To run a new comparison, use a **new output directory** after preparing the knowledge base:

```bash
python evaluation/run_evaluation.py --run-dir evaluation/runs/my-new-run
```

The runner passes only question text to the bot. It does not load the answer key. The recorded review annotations and summary scripts are tied to run01; they are not a general automatic grader for new outputs. Future runs need separate answer review.

If this dataset guides prompt/retrieval/routing changes, treat it as development data and create a fresh final holdout. Do not overwrite the frozen labels or report tuned results as an unseen benchmark.

## Files

- `src/cicd_support_rag/`: ingestion, chunking, embeddings, retrieval, LangGraph bot, providers, grounding, and UI server.
- `src/cicd_support_rag/web/`: local chatbot HTML, CSS, and JavaScript.
- `tests/`: regression checks.
- `evaluation/`: golden labels, captured runs, and metric/review tools.
- `project-config.json`: corpus, retrieval, and generation configuration.
- `.env.example`: credential/configuration template, with empty keys.
- `requirements.lock`: package versions captured from the tested local environment.

## Current limitations

Multi-part answers can omit prerequisites, use a source from the wrong workflow, or escalate despite sufficient evidence. Clarification questions can miss important decision inputs. The source-caution renderer sometimes includes unrelated warnings. Long code blocks or procedures may cross chunk boundaries. Evidence gating remains provisional and confidence is not calibrated.
