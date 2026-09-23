# AI Knowledge Base — Personal RAG Portfolio

A personal **AI knowledge base** built on your own notes. Point it at a folder of Markdown notes (portfolio skills, projects, tooling), and it builds a searchable semantic index you can ask questions against — no folders left unindexed.

It's a single FastAPI app (no separate frontend) that:

1. **Indexes** every `.md` note in a folder (`Portfolio/**/*.md`), splits it into chunks, embeds each chunk, and stores it in [Qdrant](https://qdrant.tech).
2. **Auto-re-ingests** when you add, edit, or delete a note — a file watcher detects the change and only re-embeds the files that actually changed.
3. **Answers** questions using retrieval-augmented generation (RAG): finds the most relevant note chunks and lets a Groq-hosted LLM answer from that context only.
4. **Remembers what it can't do** — technologies that aren't in the knowledge base are reported as "not part of the stack" instead of being hallucinated.

---

## Tech Stack

| Layer | Choice |
|---|---|
| API framework | [FastAPI](https://fastapi.tiangolo.com) |
| LLM | [Groq](https://groq.com) · `ChatGroq` (`openai/gpt-oss-20b`) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (384-dims) via `HuggingFaceEmbeddings` |
| Vector store | [Qdrant](https://qdrant.tech) — collection `knowledge_base` |
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

`/ask` retrieves the `TOP_K` most relevant chunks, filters out any below a relevance-score threshold, and answers strictly from that context. Capability questions are special-cased:

* "can Khaled make a `react` project" → **Yes** (React is in the KB).
* "can Khaled make a `java` project" → **No** — Java isn't in the knowledge base.

A small capability detector looks for technology names mentioned in a "can X do Y?" question, cross-checks them against the notes, and answers "No / not in the stack" for anything missing — instead of the RAG chain confidently guessing.

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
| `POST` | `/ask` | `{"question": "..."}` | RAG answer with `sources:` and `answer:` |

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

Generated at runtime and git-ignored: `.venv/`, `.vector_cache/`, `.ingest_manifest.json`, `__pycache__/`, `.idea/`.

---

## Notes & Caveats

* The knowledge base answers **only** from what's in your notes. It won't "know" anything you haven't written down, and it reports missing technologies as such rather than guessing.
* `HF_HUB_OFFLINE=1` is set so the embedding model loads only from the local cache (no startup network chatter). To enable rate-limited model downloads for a fresh machine, unset it.
* A stale clone should `git fetch origin && git reset --hard origin/main` after the history rewrite/force-push that removed `.env` from history.
