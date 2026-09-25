import os
import json
import hashlib
import re
import sqlite3
import threading
import uuid
from pathlib import Path
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import yaml

from fastapi import FastAPI
from pydantic import BaseModel

from langchain_groq import ChatGroq
from langchain_qdrant import QdrantVectorStore
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from operator import add
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    Filter,
    FieldCondition,
    MatchAny,
    FilterSelector,
    PointStruct,
)
from watchfiles import watch

load_dotenv()
os.environ.setdefault("HF_HUB_OFFLINE", "1")  # model is cached locally; no network check at startup

# -----------------------------
# Config
# -----------------------------
PORTFOLIO_PATH = "./Portfolio"          # your notes
COLLECTION_NAME = "knowledge_base"
QDRANT_URL = "http://localhost:6333"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_SIZE = 384
TOP_K = 4
SCORE_THRESHOLD = 0.5      # dense gate for the candidate POOL; final order is the hybrid rerank
RETRIEVE_K = 24             # candidate pool before the hybrid lexical rerank (TOP_K survive)
LEXICAL_BODY_BOOST = 0.03    # relevance added per question token found in the chunk text
LEXICAL_PATH_BOOST = 0.09    # relevance added per question token found in the source path (folder/file = high signal)
LEXICAL_META_BOOST = 0.05    # relevance added per question token found in frontmatter (category/summary/tech)
DIRECT_ANSWER_SCORE = 0.90   # top chunk this confident -> answer directly from the KB, no LLM
QA_HIT_SCORE = 0.82          # a stored Q->A pair this confident -> reuse it, no LLM
QA_COLLECTION = "qa_memory"  # keeps generated Q->A pairs separate from the raw notes
PIPELINE_VERSION = "rag-hybrid-v1"  # bumped when retrieval/LLM behaviour changes -> stale cached answers are ignored
MANIFEST_PATH = Path(".ingest_manifest.json")
EMBED_CACHE_DIR = Path(".vector_cache")  # stored embeddings, reused before re-embedding
SYNC_LOCK = threading.Lock()

# -----------------------------
# LLM
# -----------------------------
llm = ChatGroq(
    model="openai/gpt-oss-20b",   # or llama-3.1-8b-instant if available
    temperature=0
)

# -----------------------------
# Embeddings (lazy: model only loads when a chunk really needs embedding)
# -----------------------------
EMBEDDINGS_LOCK = threading.Lock()
_embeddings_instance = None

def get_embeddings():
    global _embeddings_instance
    if _embeddings_instance is None:
        with EMBEDDINGS_LOCK:
            if _embeddings_instance is None:
                _embeddings_instance = HuggingFaceEmbeddings(
                    model_name=EMBEDDING_MODEL,
                    encode_kwargs={"batch_size": 32},
                )
                print(f"Embedding model loaded: {EMBEDDING_MODEL}")
    return _embeddings_instance

client = QdrantClient(url=QDRANT_URL)

def get_vector_store() -> QdrantVectorStore:
    return QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=get_embeddings(),
        distance=Distance.COSINE,
    )

# -----------------------------
# QA memory (stored Q->A pairs, reused before any LLM call)
# -----------------------------
def ensure_qa_collection() -> None:
    """Create the qa_memory collection if it doesn't exist yet."""
    try:
        client.get_collection(QA_COLLECTION)
    except Exception:
        client.create_collection(
            collection_name=QA_COLLECTION,
            vectors_config=VectorParams(size=EMBEDDING_SIZE, distance=Distance.COSINE),
        )

def get_qa_store() -> QdrantVectorStore:
    ensure_qa_collection()
    return QdrantVectorStore(
        client=client,
        collection_name=QA_COLLECTION,
        embedding=get_embeddings(),
        distance=Distance.COSINE,
    )

def search_qa_memory(question: str):
    """Look for a previously-generated Q->A pair. Returns (answer, sources, score) or None."""
    try:
        vector = get_embeddings().embed_query(question)
        res = client.query_points(
            collection_name=QA_COLLECTION,
            query=vector,
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
    except Exception:
        return None
    if not res or not res.points:
        return None
    hit = res.points[0]
    score = hit.score
    if score < QA_HIT_SCORE:
        return None
    payload = hit.payload or {}
    if payload.get("pipeline") != PIPELINE_VERSION:
        return None  # stored with an older behaviour -> ignore, let the pipeline recompute
    return (
        payload.get("answer"),
        payload.get("sources", []),
        round(score, 3),
    )

# -----------------------------------
# Local exact-key Q->A cache (SQLite). THE guarantee that the same question
# string never re-hits the LLM -- independent of any embedding/vector score.
# Complements qa_memory (vector space) for near-duplicate questions.
# -----------------------------------
LOCAL_QA_DB = Path("_local_qa_cache.sqlite3")
_LOCAL_QA_LOCK = threading.RLock()
QA_STATUSES = ("missing", "local_hit", "qa_hit", "direct", "llm", "none", "error")

def _qa_cache_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(LOCAL_QA_DB, check_same_thread=False)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS qa_cache ("
        "qkey TEXT PRIMARY KEY, question TEXT, answer TEXT, sources TEXT, "
        "status TEXT, hits INTEGER NOT NULL DEFAULT 0, "
        "created_at TEXT, last_accessed_at TEXT)"
    )
    cols = {r[1] for r in conn.execute("PRAGMA table_info(qa_cache)")}
    if "status" not in cols:
        conn.execute("ALTER TABLE qa_cache ADD COLUMN status TEXT")
    if "hits" not in cols:
        conn.execute("ALTER TABLE qa_cache ADD COLUMN hits INTEGER NOT NULL DEFAULT 0")
    if "last_accessed_at" not in cols:
        conn.execute("ALTER TABLE qa_cache ADD COLUMN last_accessed_at TEXT")
    conn.execute(f"DELETE FROM qa_cache WHERE qkey NOT LIKE '{PIPELINE_VERSION}:%'")
    conn.commit()
    return conn

def _qa_key(question: str) -> str:
    norm = re.sub(r"\s+", " ", question.strip().lower())
    return f"{PIPELINE_VERSION}:{norm}"

def local_qa_lookup(question: str):
    """Exact (normalized) question -> (answer, sources, status, hits) or None. Tracks reuse."""
    try:
        with _LOCAL_QA_LOCK:
            conn = _qa_cache_conn()
            try:
                row = conn.execute(
                    "SELECT answer, sources, status, hits FROM qa_cache WHERE qkey = ?",
                    (_qa_key(question),),
                ).fetchone()
                if row:
                    conn.execute(
                        "UPDATE qa_cache SET hits = hits + 1, last_accessed_at = datetime('now') "
                        "WHERE qkey = ?",
                        (_qa_key(question),),
                    )
                    conn.commit()
            finally:
                conn.close()
        if row:
            return row[0], json.loads(row[1] or "[]"), row[2] or "unknown", row[3] + 1
    except Exception as e:
        print(f"Local QA cache read failed: {e}")
    return None

def local_qa_store(question: str, answer: str, sources: list, status: str = "llm") -> None:
    """Persist an exact Q->A so an identical re-ask skips retrieval + LLM entirely."""
    try:
        with _LOCAL_QA_LOCK:
            conn = _qa_cache_conn()
            try:
                conn.execute(
                    "INSERT INTO qa_cache "
                    "(qkey, question, answer, sources, status, hits, created_at, last_accessed_at) "
                    "VALUES (?,?,?,?,?,1,datetime('now'),datetime('now')) "
                    "ON CONFLICT(qkey) DO UPDATE SET "
                    "answer=excluded.answer, sources=excluded.sources, status=excluded.status, "
                    "hits=hits+1, last_accessed_at=datetime('now')",
                    (_qa_key(question), question.strip(), answer,
                     json.dumps(sources, ensure_ascii=False), status),
                )
                conn.commit()
            finally:
                conn.close()
    except Exception as e:
        print(f"Local QA cache write failed: {e}")

def store_qa_pair(question: str, answer: str, sources: list) -> None:
    """Persist a generated Q->A so the next similar question skips the LLM."""
    try:
        vector = get_embeddings().embed_query(question)
        qid = uuid.uuid5(uuid.NAMESPACE_OID, f"qa#{question.strip().lower()}")
        client.upsert(
            collection_name=QA_COLLECTION,
            points=[PointStruct(
                id=qid,
                vector=vector,
                payload={
                    "question": question.strip(),
                    "answer": answer,
                    "sources": sources,
                    "kind": "qa_memory",
                    "pipeline": PIPELINE_VERSION,
                },
            )],
        )
    except Exception as e:
        print(f"QA store failed: {e}")

def ingest_qa_once() -> None:
    """Ensure every ingested note also gets a stored Q->A baseline (optional warm-up)."""
    ensure_qa_collection()

# -----------------------------
# Notes & manifest
# -----------------------------
def list_note_files() -> list[Path]:
    """All .md notes, sorted so ingest order is deterministic. Files starting with '_'
    (templates, guides, scratch) are ignored so they never pollute the knowledge base."""
    files = [p for p in Path(PORTFOLIO_PATH).rglob("*.md") if not p.name.startswith("_")]
    return sorted(files, key=lambda p: str(p).lower())

def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def current_hashes(files: list[Path] | None = None) -> dict:
    files = files if files is not None else list_note_files()
    return {str(f): file_hash(f) for f in files}

def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {}

def save_manifest(manifest: dict) -> None:
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

# -----------------------------
# Embedding cache (stored on disk, reused first)
# -----------------------------
def _safe_name(source: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", source)

def _cache_path(source: str, h: str) -> Path:
    return EMBED_CACHE_DIR / f"{h[:16]}-{_safe_name(source)}.json"

def _load_cached(source: str, h: str) -> list | None:
    p = _cache_path(source, h)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    if data.get("file") != source or data.get("hash") != h:
        return None
    return data.get("chunks")

def _save_cache(source: str, h: str, chunks: list) -> None:
    EMBED_CACHE_DIR.mkdir(exist_ok=True)
    _cache_path(source, h).write_text(
        json.dumps({"file": source, "hash": h, "chunks": chunks}, ensure_ascii=False),
        encoding="utf-8",
    )

def _prune_caches(current: dict) -> None:
    """Remove cache files that no longer match the current manifest."""
    try:
        expected = {_cache_path(p, h).name for p, h in current.items()}
        for c in EMBED_CACHE_DIR.glob("*.json"):
            if c.name not in expected:
                c.unlink()
    except Exception as e:
        print(f"Cache prune failed: {e}")

# -----------------------------
# Splitting & embedding (cache-first)
# -----------------------------
# Canonical note skeleton: YAML frontmatter gives every chunk searchable,
# high-signal metadata (category / summary / tech) that the hybrid rerank uses.
CATEGORY_FROM_DIR = {
    "Backend": "backend",
    "Frontend": "frontend",
    "Data Science": "data-science",
    "DevOps": "devops",
    "AI": "ai",
    "ML": "ml",
    "Projects": "project",
}

def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Extract the YAML frontmatter block -> (meta dict, original text). Never raises."""
    meta: dict = {}
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, flags=re.S)
    if not m:
        return meta, text
    try:
        parsed = yaml.safe_load(m.group(1))
        if isinstance(parsed, dict):
            meta = parsed
    except Exception as e:
        print(f"Frontmatter parse failed: {e}")
    return meta, text

def _category_for(path: Path, meta: dict) -> str:
    if isinstance(meta.get("category"), str) and meta["category"].strip():
        return meta["category"].strip().lower()
    # Portfolio/<TopLevelFolder>/... (projects nest one level deeper)
    parts = path.parts
    top = parts[1] if len(parts) >= 3 else (parts[-2] if len(parts) >= 2 else "")
    return CATEGORY_FROM_DIR.get(top, "other")

def _tech_for(meta: dict) -> list[str]:
    """tech from the frontmatter `tech:` list or `tags:` entries (uses/<tech>, skill/<name>)."""
    tech: list[str] = []
    raw = meta.get("tech")
    if isinstance(raw, list):
        tech.extend(str(t).strip() for t in raw)
    elif isinstance(raw, str) and raw.strip():
        tech.extend(re.split(r"[\s,]+", raw.strip()))
    for tag in meta.get("tags") or []:
        s = str(tag).strip() if isinstance(tag, str) else ""
        if s.startswith("uses/") or s.startswith("skill/"):
            tech.append(s.split("/", 1)[1])
    return sorted({t for t in tech if t})

def _split_file(file: Path):
    try:
        docs = TextLoader(str(file), encoding="utf-8").load()
    except Exception as e:
        print(f"Skipping {file}: {e}")
        return []

    meta, text = _parse_frontmatter(docs[0].page_content)
    category = _category_for(file, meta)
    tech = _tech_for(meta)
    summary = meta.get("summary")

    splits = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150).split_documents(docs)
    for d in splits:
        d.metadata["name"] = Path(d.metadata.get("source", "")).stem
        d.metadata["source"] = str(file)
        d.metadata["category"] = category
        d.metadata["tech"] = tech
        if isinstance(summary, str) and summary.strip():
            d.metadata["summary"] = summary.strip()
    return splits

def _chunk_metadata_for(file: Path) -> dict:
    """Frontmatter-derived metadata (category/tech/summary) tied to a file's chunks."""
    meta, _ = _parse_frontmatter(file.read_text(encoding="utf-8"))
    chunk_meta = {"category": _category_for(file, meta), "tech": _tech_for(meta)}
    summ = meta.get("summary")
    if isinstance(summ, str) and summ.strip():
        chunk_meta["summary"] = summ.strip()
    return chunk_meta

def build_points(files: list[Path]):
    """Return (points, newly_embedded_count). Reuses stored embeddings from cache."""
    points = []
    embedded = 0

    for f in files:
        source = str(f)
        h = file_hash(f)

        cached = _load_cached(source, h)
        if cached is not None:
            # cached chunks predate the frontmatter metadata -> enrich in place so the
            # hybrid rerank gets category/summary/tech without re-embedding
            extra = _chunk_metadata_for(f)
            for e in cached:
                e["metadata"].update(extra)
            points.extend(cached)
            continue

        splits = _split_file(f)
        if not splits:
            continue

        vectors = get_embeddings().embed_documents([d.page_content for d in splits])
        entries = []
        for i, (d, v) in enumerate(zip(splits, vectors)):
            eid = str(uuid.uuid5(uuid.NAMESPACE_OID, f"{source}#{i}"))
            entries.append({
                "id": eid,
                "page_content": d.page_content,
                "metadata": d.metadata,
                "vector": v,
            })
            points.append(entries[-1])

        _save_cache(source, h, entries)
        embedded += len(entries)

    return points, embedded

def _upsert_points(points: list[dict]) -> None:
    if not points:
        return
    batch = 64
    for i in range(0, len(points), batch):
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                PointStruct(
                    id=p["id"],
                    vector=p["vector"],
                    payload={"page_content": p["page_content"], "metadata": p["metadata"]},
                )
                for p in points[i:i + batch]
            ],
        )

def delete_points_for_sources(sources: list[str]) -> None:
    if not sources:
        return
    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=FilterSelector(
            filter=Filter(
                must=[FieldCondition(key="metadata.source", match=MatchAny(any=sources))]
            )
        ),
    )

# -----------------------------
# 1. Ingestion
# -----------------------------
def ingest_documents():
    """Full rebuild, but reuses stored embeddings for unchanged notes."""
    with SYNC_LOCK:
        files = list_note_files()
        print(f"Loading documents (sorted)... {len(files)} files")

        points, embedded = build_points(files)

        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_SIZE, distance=Distance.COSINE),
        )
        _upsert_points(points)

        current = current_hashes(files)
        save_manifest(current)
        _prune_caches(current)
        print(f"Stored {len(points)} chunks (newly embedded: {embedded})")
        return len(points)

def sync_documents():
    """Incremental: only re-embed new/changed files, drop removed ones."""
    with SYNC_LOCK:
        files = list_note_files()
        current = current_hashes(files)
        manifest = load_manifest()

        changed = [p for p, h in current.items() if manifest.get(p) != h]
        removed = [p for p in manifest if p not in current]

        if not changed and not removed:
            return {"status": "ok", "changed": 0, "removed": 0, "chunks_added": 0}

        delete_points_for_sources(changed + removed)

        points, embedded = build_points([Path(p) for p in changed])
        _upsert_points(points)

        save_manifest(current)
        _prune_caches(current)
        print(f"Sync: {len(changed)} changed, {len(removed)} removed, {embedded} newly embedded, {len(points)} chunks upserted")
        return {"status": "ok", "changed": len(changed), "removed": len(removed), "chunks_added": len(points)}

# -----------------------------
# File watcher (auto re-ingest)
# -----------------------------
def _watch_loop():
    try:
        sync_documents()
    except Exception as e:
        print(f"Startup sync failed: {e}")

    print(f"Watching {PORTFOLIO_PATH} for new/edited files...")
    try:
        for changes in watch(PORTFOLIO_PATH, debounce=1500, step=200):
            touched = sorted({p for _, p in changes})
            print(f"Detected change: {touched}")
            try:
                result = sync_documents()
                if result["changed"] or result["removed"]:
                    print(f"Auto-ingested: {result}")
            except Exception as e:
                print(f"Auto-ingest failed: {e}")
    except Exception as e:
        print(f"Watcher stopped: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=_watch_loop, daemon=True, name="portfolio-watcher").start()
    yield

# -----------------------------
# FastAPI
# -----------------------------
app = FastAPI(title="AI Knowledge Base - Multi Agent", lifespan=lifespan)

class Question(BaseModel):
    question: str

# -----------------------------
# Capability check: tech not in the KB means Khaled can't do it
# -----------------------------
CAPABILITY_RE = re.compile(r"\b(can|could|does|do|able|make|build|work|know|stack|skill)\b")

STOPWORDS = set("""
a an the and or of to for in on at is are am be was were can could does do did
make makes build builds work works know knows use uses used using with from this
that these those i you your my mine me us we our it its what who when where why
how about not no yes project projects as by him his her khaled app apps site web
service services platform system tool framework language engine api application
program code frontend backend database db something anything thing things someone
anybody some any many much more most very really just also one two three website
websites webpage webapp idea ideas product products example examples sample samples
stuff job jobs business company feature features
""".split())

# Lighter stopword set used ONLY for the retrieval keyword boost. Generic function
# words are dropped, but topic words such as "frontend", "framework", "database",
# "docker" are kept so short notes (e.g. React.md) are not buried by hub pages.
RETRIEVAL_STOPWORDS = frozenset("""
a an the and or of to for in on at is are am be was were can could do does did
what who when where why how i you your my mine me us we our it its they them
he she his her khaled not no yes about over under with without from also then
than so just very really more most even still
""".split())

def vault_text() -> str:
    parts = []
    for f in list_note_files():
        try:
            parts.append(f.read_text(encoding="utf-8"))
        except Exception:
            pass
    return " ".join(parts).lower()

def missing_techs(question: str) -> list[str]:
    """Technologies named in a capability question that are absent from the KB."""
    q = question.lower()
    if not CAPABILITY_RE.search(q):
        return []
    vt = vault_text()
    tokens = re.findall(r"[a-z0-9+#.-]+", q)
    missing = []
    for t in tokens:
        if t in STOPWORDS or re.search(rf"\b{re.escape(t)}\b", vt):
            continue
        missing.append(t)
    return sorted(set(missing))

# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def home():
    return {"message": "AI Knowledge Base Multi-Agent is running"}

@app.post("/ingest")
def run_ingest():
    chunks = ingest_documents()
    return {"status": "ok", "chunks": chunks}

@app.post("/sync")
def run_sync():
    return sync_documents()

# -----------------------------
# LangGraph: /ask pipeline (compiled StateGraph)
# -----------------------------
class AskState(TypedDict, total=False):
    question: str
    answer: str
    sources: list
    missing: list
    status: str            # gate -> local_cache -> qa_memory -> retrieve -> direct | llm
    retrieved: list        # [(Document, score)] surviving thresholding
    top_score: float

def _ask_gate_node(state: AskState) -> dict:
    """0. Capability gate: a named tech absent from the KB -> immediate No, no retrieval, no LLM."""
    missing = missing_techs(state["question"])
    if missing:
        names = ", ".join(missing)
        if len(missing) == 1:
            answer = f"No - Khaled does not work with {missing[0]}. It is not in the knowledge base."
        else:
            answer = f"No - Khaled does not work with {names}. None of them are in the knowledge base."
        local_qa_store(state["question"], answer, [], "missing")
        print(f"Capability gate: {names} missing -> returned No, skipped retrieval + LLM")
        return {"answer": answer, "sources": [], "missing": missing, "status": "missing"}
    return {"missing": [], "status": "ok"}

def _local_cache_node(state: AskState) -> dict:
    """1. Exact normalized match in the local SQLite cache -> cached answer, no LLM/retrieval."""
    if state["status"] == "missing":
        return {}
    hit = local_qa_lookup(state["question"])
    if hit:
        answer, sources, status, hits = hit
        print(f"Local QA cache hit (status={status}, hits={hits}): reused stored answer, no LLM")
        return {"answer": answer, "sources": sources, "status": "local_hit"}
    return {"status": "ok"}

def _qa_memory_node(state: AskState) -> dict:
    """2. Reuse a stored Q->A pair (score >= QA_HIT_SCORE) -> no LLM, no re-query."""
    if state["status"] in ("missing", "local_hit"):
        return {}
    hit = search_qa_memory(state["question"])
    if hit:
        qa_answer, qa_sources, qa_score = hit
        local_qa_store(state["question"], qa_answer, qa_sources, "qa_hit")
        print(f"QA memory hit (score={qa_score}): reusing stored answer, no LLM")
        return {"answer": qa_answer, "sources": qa_sources, "status": "qa_hit"}
    return {"status": "ok"}

def _question_tokens(question: str) -> list[str]:
    """Content words of a question (minimal stopword filter) used for the keyword boost."""
    toks = re.findall(r"[a-z0-9+#.-]+", question.lower())
    return [t for t in toks if len(t) > 1 and t not in RETRIEVAL_STOPWORDS]

def _hybrid_score(doc, dense: float, tokens: list[str]) -> float:
    """dense similarity + keyword hits. Path hits (file name/folder) and frontmatter
    metadata (category/summary/tech) are the strongest signals for short notes that a
    generic embedding buries (e.g. React.md vs a 'frontend framework?' question).
    The dense score is preserved separately, so the direct-answer gate still uses real cosine."""
    text = doc.page_content.lower()
    path = str(doc.metadata.get("source") or "").lower()
    meta = doc.metadata or {}
    meta_text = " ".join([
        str(meta.get("category") or ""),
        str(meta.get("summary") or ""),
        " ".join(meta.get("tech") or []),
    ]).lower()
    body = sum(1 for t in tokens if re.search(rf"\b{re.escape(t)}\b", text))
    path_m = sum(1 for t in tokens if re.search(rf"\b{re.escape(t)}\b", path))
    meta_m = sum(1 for t in tokens if re.search(rf"\b{re.escape(t)}\b", meta_text))
    # Not capped at 1.0: it is only an ordering key, the true cosine score rides along.
    return dense + LEXICAL_BODY_BOOST * body + LEXICAL_PATH_BOOST * path_m + LEXICAL_META_BOOST * meta_m

def _retrieve_node(state: AskState) -> dict:
    """3. Retrieve from the KB (Qdrant). Rerank the pool with a keyword boost, keep TOP_K."""
    if state["status"] in ("missing", "local_hit", "qa_hit"):
        return {}
    try:
        raw = get_vector_store().similarity_search_with_relevance_scores(state["question"], k=RETRIEVE_K)
    except Exception as e:
        print(f"Retrieval failed: {e}")
        local_qa_store(state["question"], f"Retrieval failed: {e}", [], "error")
        return {"answer": f"Retrieval failed: {e}", "sources": [], "status": "error"}
    candidates = [(d, s) for d, s in raw if s >= SCORE_THRESHOLD]
    if not candidates:
        local_qa_store(state["question"],
                       "No relevant documents found in the knowledge base. Run /ingest first.",
                       [], "none")
        return {"answer": "No relevant documents found in the knowledge base. Run /ingest first.", "sources": [], "status": "none", "retrieved": []}
    tokens = _question_tokens(state["question"])
    candidates.sort(key=lambda p: _hybrid_score(p[0], p[1], tokens), reverse=True)
    # de-duplicate so the same source file can't occupy several slots
    seen: set[str] = set()
    unique = []
    for d, s in candidates:
        src = str(d.metadata.get("source") or "")
        if src in seen:
            continue
        seen.add(src)
        unique.append((d, s))
    return {"retrieved": unique[:TOP_K], "top_score": unique[0][1], "status": "ok"}

def _direct_answer_node(state: AskState) -> dict:
    """4a. Top chunk confident enough (>= DIRECT_ANSWER_SCORE) -> answer straight from KB, no LLM."""
    top_doc, top_score = state["retrieved"][0]
    answer = top_doc.page_content.strip()
    sources = [{
        "name": top_doc.metadata.get("name"),
        "source": top_doc.metadata.get("source"),
        "score": round(top_score, 3),
    }]
    local_qa_store(state["question"], answer, sources, "direct")
    store_qa_pair(state["question"], answer, sources)
    print(f"Direct answer (score={top_score:.3f}): returned chunk, skipped LLM")
    return {"answer": answer, "sources": sources, "status": "direct"}

def _llm_answer_node(state: AskState) -> dict:
    """4b. Otherwise synthesize with the LLM, then store the pair for future reuse."""
    context = "\n\n".join(d.page_content for d, _ in state["retrieved"])
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant for Khaled's personal knowledge base. "
                   "Everything covered in the context below is part of his stack and skills. "
                   "Answer the question using the context, and answer directly and confidently. "
                   "When asked about his preferred tool or framework, name the technology from the "
                   "context that fits best even if the word 'preferred' does not appear in the notes. "
                   "Only say you don't know if nothing in the context relates."),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ])
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": state["question"]})
    sources = [
        {
            "name": d.metadata.get("name"),
            "source": d.metadata.get("source"),
            "score": round(s, 3),
        }
        for d, s in state["retrieved"]
    ]
    local_qa_store(state["question"], answer, sources, "llm")
    store_qa_pair(state["question"], answer, sources)
    print(f"LLM answer stored in local cache + QA memory ({len(sources)} sources)")
    return {"answer": answer, "sources": sources, "status": "llm"}

def _route_after_gate(state: AskState) -> str:
    return END if state.get("status") == "missing" else "local_cache"

def _route_after_local_cache(state: AskState) -> str:
    return END if state.get("status") == "local_hit" else "qa_memory"

def _route_after_qa_memory(state: AskState) -> str:
    return END if state.get("status") == "qa_hit" else "retrieve"

def _route_after_retrieve(state: AskState) -> str:
    status = state.get("status")
    if status in ("error", "none"):
        return END
    return "direct_answer" if state["top_score"] >= DIRECT_ANSWER_SCORE else "llm_answer"

def build_ask_graph():
    """Compile the /ask flow: gate -> local_cache -> qa_memory -> retrieve -> (direct | llm)."""
    graph = StateGraph(AskState)
    graph.add_node("gate", _ask_gate_node)
    graph.add_node("local_cache", _local_cache_node)
    graph.add_node("qa_memory", _qa_memory_node)
    graph.add_node("retrieve", _retrieve_node)
    graph.add_node("direct_answer", _direct_answer_node)
    graph.add_node("llm_answer", _llm_answer_node)
    graph.add_edge(START, "gate")
    graph.add_conditional_edges("gate", _route_after_gate)
    graph.add_conditional_edges("local_cache", _route_after_local_cache)
    graph.add_conditional_edges("qa_memory", _route_after_qa_memory)
    graph.add_conditional_edges("retrieve", _route_after_retrieve)
    graph.add_edge("direct_answer", END)
    graph.add_edge("llm_answer", END)
    return graph.compile()

ask_graph = build_ask_graph()
# -----------------------------
# /ask route -> compiled LangGraph StateGraph
# -----------------------------
@app.post("/ask")
def ask_question(body: Question):
    """Run the compiled LangGraph: capability gate -> local cache -> QA memory -> retrieve -> (direct | LLM)."""
    result = ask_graph.invoke({"question": body.question})
    return {"answer": result["answer"], "sources": result.get("sources", [])}

# -----------------------------
# Local QA cache: list + analysis of stored answers
# -----------------------------
@app.get("/cache")
def cache_list():
    """All locally-cached Q->A pairs, most-reused first."""
    with _LOCAL_QA_LOCK:
        conn = _qa_cache_conn()
        try:
            rows = conn.execute(
                "SELECT question, answer, sources, status, hits, created_at, last_accessed_at "
                "FROM qa_cache ORDER BY hits DESC, created_at DESC"
            ).fetchall()
        finally:
            conn.close()
    return [
        {
            "question": r[0],
            "answer": r[1],
            "sources": json.loads(r[2] or "[]"),
            "status": r[3],
            "hits": r[4],
            "created_at": r[5],
            "last_accessed_at": r[6],
        }
        for r in rows
    ]

@app.get("/cache/analysis")
def cache_analysis():
    """Aggregate stats over the local cache: answer mix, reuse, top questions/sources."""
    rows = cache_list()
    if not rows:
        return {"total": 0, "total_hits": 0, "by_status": {}, "avg_answer_chars": 0,
                "top_questions": [], "top_sources": []}

    by_status = {}
    source_counts = {}
    total_hits = 0
    total_chars = 0
    for r in rows:
        status = r["status"] or "unknown"
        by_status[status] = by_status.get(status, 0) + 1
        total_hits += r["hits"]
        total_chars += len(r["answer"])
        for s in r["sources"]:
            src = s.get("source") or "unknown"
            source_counts[src] = source_counts.get(src, 0) + 1

    top_questions = sorted(
        [{"question": r["question"], "status": r["status"], "hits": r["hits"],
          "created_at": r["created_at"]} for r in rows],
        key=lambda x: x["hits"],
        reverse=True,
    )[:10]

    top_sources = sorted(
        [{"source": s, "used": c} for s, c in source_counts.items()],
        key=lambda x: x["used"],
        reverse=True,
    )[:10]

    return {
        "total": len(rows),
        "total_hits": total_hits,
        "by_status": dict(sorted(by_status.items())),
        "avg_answer_chars": round(total_chars / len(rows), 1),
        "top_questions": top_questions,
        "top_sources": top_sources,
    }

@app.delete("/cache")
def cache_clear():
    """Wipe the local Q->A cache."""
    with _LOCAL_QA_LOCK:
        conn = _qa_cache_conn()
        try:
            conn.execute("DELETE FROM qa_cache")
            conn.commit()
        finally:
            conn.close()
    return {"status": "ok", "cleared": True}
