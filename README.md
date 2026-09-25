# AI Knowledge Base — Personal RAG Portfolio

A personal **AI knowledge base** built on your own notes. Point it at a folder of Markdown notes (portfolio skills, projects, tooling), and it builds a searchable semantic index you can ask questions against — no folders left unindexed.

It's a single FastAPI app (no separate frontend) that:

1. **Indexes** every `.md` note in a folder (`Portfolio/**/*.md`), splits it into chunks, embeds each chunk, and stores it in [Qdrant](https://qdrant.tech).
2. **Auto-re-ingests** when you add, edit, or delete a note — a file watcher detects the change and only re-embeds the files that actually changed.
3. **Answers** questions using retrieval-augmented generation (RAG): finds the most relevant note chunks and lets a Groq-hosted LLM answer from that context only.
4. **Never asks the LLM the same question twice** — every question→answer pair is cached in a local SQLite database, so an exact re-ask is answered instantly with no retrieval, no Qdrant, and no LLM call (a vector `qa_memory` collection catches near-duplicate re-asks).
5. **Analyzes cached answers** — `/cache/analysis` reports the answer mix (direct / LLM / missing...), reuse counts, and top questions and sources.
6. **Remembers what it can't do** — technologies that aren't in the knowledge base are reported as "not part of the stack" instead of being hallucinated.

---

## Tech Stack

| Layer | Choice |
|---|---|
| API framework | [FastAPI](https://fastapi.tiangolo.com) |
| LLM | [Groq](https://groq.com) · `ChatGroq` (`openai/gpt-oss-20b`) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (384-dims) via `HuggingFaceEmbeddings` |
| Vector store | [Qdrant](https://qdrant.tech) — collections `knowledge_base` + `qa_memory` |
| Local QA cache | SQLite (`_local_qa_cache.sqlite3`) — exact question → stored answer |
| Orchestration | [LangGraph](https://langchain-ai.github.io/langgraph/) — `/ask` is a compiled `StateGraph` (`gate → local_cache → qa_memory → retrieve → direct|llm`) |
| Markdown loading | `DirectoryLoader` + `TextLoader`, `RecursiveCharacterTextSplitter` |
| File watching | `watchfiles` |

---

## Architecture

```
                      Portfolio/  (your notes, .md)
                           │  watchfiles  ── on change ──┐
                           ▼                            ▼
                DirectoryLoader ──────────────┐   sync_documents()
                RecursiveCharacterTextSplitter │  (incremental, cache-first)
                           ▼                  │
                HuggingFaceEmbeddings ────────┘
                           │  vectors
                           ▼
                    Qdrant (localhost:6333)
                           ▼   similarity search
                     FastAPI  /ask
                           │  top-k chunks → context
                           ▼
                  ChatGroq  (answer from context only)
```

### Ingestion (idempotent + cached)

* `/ingest` — full rebuild: loads **all** notes (sorted), splits, embeds, stores. Safe to re-run: it recreates the collection so there are no duplicates.
* `/sync` — incremental: compares a sha256 manifest of note files against what's already stored, and only re-embeds **new/changed** files, deleting chunks for removed files. No-ops fast (~0s) when nothing changed.
* A **file watcher** runs on startup and calls the same sync automatically whenever you edit a note — the knowledge base stays current without you re-running anything.

Embedding results are cached on disk (`.vector_cache/`) **keyed by file + content hash**, so:

* re-running `/ingest` reuses cached vectors instead of re-embedding (≈5× faster, ~1.5s warm),
* only new/changed files trigger the embedding model, and only their embeddings are recomputed,
* deleted files are pruned from the cache automatically.

This is why startup is model-free: the embedding model loads **lazily**, only when a chunk actually needs embedding (a new/edited file, or the first `/ask`).

### Asking questions

`/ask` runs a LangGraph pipeline: **capability gate → local SQLite cache → QA memory → retrieve → (direct | LLM)**. It short-circuits as early as possible:

* **Capability gate** — a named tech absent from the KB → "No" (no retrieval, no LLM).
* **Local SQLite cache** — an exactly-matching (normalized) question → reuses the stored answer instantly. This is what guarantees the same question never hits the LLM twice.
* **QA memory** — a near-duplicate question matching a stored Q→A pair above a cosine threshold → reuses that answer (no LLM, no re-query).
* **Retrieve** — the top chunk at ≥ `0.90` relevance → answered directly from the KB (no LLM); otherwise the LLM synthesizes an answer from the retrieved chunks only.

Every terminal answer (including "No" and "no relevant documents") is persisted in the local cache, so repeated asks short-circuit without touching Qdrant or Groq.

Inspecting and analyzing the cache:

* `GET /cache` — all stored Q→A pairs, most-reused first.
* `GET /cache/analysis` — stats: total pairs, total reuses, breakdown by answer type (`llm`, `direct`, `missing`...), average answer length, top questions, top sources.
* `DELETE /cache` — wipe the local cache.

A small capability detector looks for technology names mentioned in a "can X do Y?" question, cross-checks them against the notes, and answers "No / not in the stack" for anything missing — instead of the RAG chain confidently guessing.

### LangGraph pipeline

`/ask` is a compiled [LangGraph](https://langchain-ai.github.io/langgraph/) state graph
(`main.py: build_ask_graph`). Each node reads/writes the shared `AskState`
(`question`, `answer`, `sources`, `missing`, `retrieved`, `top_score`, `status`) and
returns a partial update; conditional edges short-circuit as early as possible.

```
 START
   │
   ▼
 ▢ gate ────────── missing tech ──► END      ("No, not in the stack", no retrieval/LLM)
   │ passes
   ▼
 ▢ local_cache ── exact SQLite hit ──► END   (cached answer, no LLM)
   │ miss
   ▼
 ▢ qa_memory ── near-duplicate hit ──► END   (stored Q→A, no LLM)
   │ miss
   ▼
 ▢ retrieve ── error / no docs ──► END
   │ hits
   ├─ top_score ≥ 0.90 ──► ▢ direct_answer ──► END  (raw chunk, no LLM)
   └─ else ────────────► ▢ llm_answer ──► END       (Groq synthesis)
```

| # | Node | Short-circuits to END when… | Resulting `status` |
|---|---|---|---|
| 1 | `gate` | a named tech in the question is absent from the KB | `missing` |
| 2 | `local_cache` | exact (normalized) question found in SQLite | `local_hit` |
| 3 | `qa_memory` | vector-similar stored Q→A pair ≥ `QA_HIT_SCORE` (0.82) | `qa_hit` |
| 4 | `retrieve` | nothing above `SCORE_THRESHOLD` (0.5) / retrieval error | `none`, `error` |
| 5a | `direct_answer` | top dense score ≥ `DIRECT_ANSWER_SCORE` (0.90) — answers with the raw chunk | `direct` |
| 5b | `llm_answer` | otherwise — Groq synthesizes from the retrieved chunks | `llm` |

Tuning knobs (top of `main.py`):

* `TOP_K` / `RETRIEVE_K` — how many chunks are returned vs. pulled as candidate pool.
* `SCORE_THRESHOLD` — dense cosine gate for candidates; final order is the hybrid rerank
  (`LEXICAL_BODY_BOOST` / `LEXICAL_PATH_BOOST` / `LEXICAL_META_BOOST` keyword boosts,
  keyword matches win but true cosine is preserved for the direct-answer gate).
* `DIRECT_ANSWER_SCORE`, `QA_HIT_SCORE` — confidence gates that skip the LLM.

**Extending:** add a node with `graph.add_node("name", fn)`, then a conditional edge
(`graph.add_conditional_edges("parent", router_fn)`) that returns the next node name or
`END`; set its route on `AskState` so it behaves like the nodes above.

Both answer nodes persist the Q→A pair (SQLite + `qa_memory`), so every path is cheap on a re-ask.

### Note skeleton (frontmatter schema)

Every note follows a small YAML frontmatter template. It is *the* thing that makes the
retrieval good: `category`, `summary`, and `tech` are extracted at ingest time, stored on
every chunk, and feed the keyword boost (`LEXICAL_META_BOOST`).

```yaml
---
title: React
category: frontend                  # backend | frontend | devops | data-science | ai | ml | project
tags:
  - skill/frontend
  - uses/react
summary: "One clear sentence that answers what this is / that Khaled uses it."
tech: [React, JavaScript]
related: ["[[Next.js]]"]           # obsidian links, not ingested
---
```

* Copy `Portfolio/_TEMPLATE.md` for new notes — it's skipped by the ingester (filenames
  starting with `_` are never indexed), so it won't pollute the knowledge base.
* `summary` is a single self-contained sentence; announcements like "Khaled uses X for Y"
  make answers reliable — the LLM answers only from note text.
* After restructuring existing notes, run `POST /ingest` once so every chunk picks up the
  new metadata (cached embeddings are reused, only payloads refresh).

---

## Getting Started

### Prerequisites

* Python 3.13
* A running [Qdrant](https://qdrant.tech/documentation/quickstart) server (`http://localhost:6333`) — the default `docker run -p 6333:6333 qdrant/qdrant` works.

### 1. Install

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
```

### 2. Configure

Copy the template and add your Groq API key (`GROQ_API_KEY`):

```bash
copy .env.example .env        # then fill in GROQ_API_KEY
```

The app reads secrets from `.env` via `python-dotenv`. **`.env` is git-ignored** — never commit it.

### 3. Run

```bash
python -m uvicorn main:app --reload --port 8000
```

Open `http://localhost:8000` (API docs at `/docs`).

### 4. Use it

* `POST /ingest` — (re)build the index from all notes.
* `POST /sync` — incremental re-index (also runs automatically on file changes).
* `POST /ask` — ask a question:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"What technologies does Khaled use?\"}"
```

---

## API Reference

All endpoints return JSON.

| Method | Path | Body | Description |
|---|---|---|---|
| `GET` | `/` | — | Health check |
| `POST` | `/ingest` | — | Full rebuild of the vector index |
| `POST` | `/sync` | — | Incremental sync (new/edited/removed notes) |
| `POST` | `/ask` | `{"question": "..."}` | Compiled LangGraph: capability gate → local cache → QA memory → retrieve → (direct \| LLM). Returns `sources:` + `answer:` |
| `GET` | `/cache` | — | List locally-cached Q→A pairs (most-reused first) |
| `GET` | `/cache/analysis` | — | Stats over cached answers |
| `DELETE` | `/cache` | — | Clear the local Q→A cache |

`/ask` response:

```json
{
  "answer": "FastAPI is a modern, high-performance Python web framework built on Starlette and Pydantic...",
  "sources": [
    {
      "name": "FastAPI",
      "source": "Portfolio\\Backend\\FastAPI.md",
      "score": 0.786
    }
  ]
}
```

---

## Project Layout

```
ai-knowledge-base/
├── main.py                  # FastAPI app: RAG, ingest, sync, /ask
├── Portfolio/               # YOUR notes (git-ignored generated files live here)
├── requirements.txt
├── .env.example             # template — copy to .env (never committed)
└── .gitignore
```

Generated at runtime and git-ignored: `.venv/`, `.vector_cache/`, `.ingest_manifest.json`, `_local_qa_cache.sqlite3`, `__pycache__/`, `.idea/`.

---

## Notes & Caveats

* The knowledge base answers **only** from what's in your notes. It won't "know" anything you haven't written down, and it reports missing technologies as such rather than guessing.
* `HF_HUB_OFFLINE=1` is set so the embedding model loads only from the local cache (no startup network chatter). To enable rate-limited model downloads for a fresh machine, unset it.
* A stale clone should `git fetch origin && git reset --hard origin/main` after the history rewrite/force-push that removed `.env` from history.
