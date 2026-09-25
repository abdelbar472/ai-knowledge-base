import hashlib
import json
import re
import threading
import uuid
from pathlib import Path

import yaml
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchAny,
    PointStruct,
    VectorParams,
)
from watchfiles import watch

from config import (
    COLLECTION_NAME,
    EMBEDDING_SIZE,
    EMBED_CACHE_DIR,
    MANIFEST_PATH,
    PORTFOLIO_PATH,
)
from embeddings import get_embeddings
from store import client

SYNC_LOCK = threading.Lock()

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
# Ingestion
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