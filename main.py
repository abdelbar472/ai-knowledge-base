import os
import json
import hashlib
import re
import threading
import uuid
from pathlib import Path
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from fastapi import FastAPI
from pydantic import BaseModel

from langchain_groq import ChatGroq
from langchain_qdrant import QdrantVectorStore
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
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
SCORE_THRESHOLD = 0.6
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
# Notes & manifest
# -----------------------------
def list_note_files() -> list[Path]:
    """All .md notes, sorted so ingest order is deterministic."""
    files = list(Path(PORTFOLIO_PATH).rglob("*.md"))
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
def _split_file(file: Path):
    try:
        docs = TextLoader(str(file), encoding="utf-8").load()
    except Exception as e:
        print(f"Skipping {file}: {e}")
        return []

    splits = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150).split_documents(docs)
    for d in splits:
        d.metadata["name"] = Path(d.metadata.get("source", "")).stem
        d.metadata["source"] = str(file)
    return splits

def build_points(files: list[Path]):
    """Return (points, newly_embedded_count). Reuses stored embeddings from cache."""
    points = []
    embedded = 0

    for f in files:
        source = str(f)
        h = file_hash(f)

        cached = _load_cached(source, h)
        if cached is not None:
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

@app.post("/ask")
def ask_question(body: Question):
    missing = missing_techs(body.question)
    if missing:
        names = ", ".join(missing)
        if len(missing) == 1:
            answer = f"No - Khaled does not work with {missing[0]}. It is not in the knowledge base."
        else:
            answer = f"No - Khaled does not work with {names}. None of them are in the knowledge base."
        return {"answer": answer, "sources": []}

    try:
        raw = get_vector_store().similarity_search_with_relevance_scores(body.question, k=TOP_K)
    except Exception as e:
        return {"answer": f"Retrieval failed: {e}", "sources": []}

    retrieved = [(d, s) for d, s in raw if s >= SCORE_THRESHOLD]

    if not retrieved:
        return {"answer": "No relevant documents found in the knowledge base. Run /ingest first.", "sources": []}

    context = "\n\n".join(d.page_content for d, _ in retrieved)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant for Khaled's personal knowledge base. Khaled owns this vault, so anything covered in the context below is part of his stack and skills. Answer the question using ONLY the following context. If the context does not answer the question, say you don't know."),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ])
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": body.question})

    return {
        "answer": answer,
        "sources": [
            {
                "name": d.metadata.get("name"),
                "source": d.metadata.get("source"),
                "score": round(s, 3),
            }
            for d, s in retrieved
        ],
    }