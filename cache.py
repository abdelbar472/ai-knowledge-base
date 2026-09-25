import json
import re
import sqlite3
import threading

from config import LOCAL_QA_DB, PIPELINE_VERSION

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


def cache_list():
    """All locally-cached Q->A pairs, most-reused first. Read-only helper for routes/shell."""
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