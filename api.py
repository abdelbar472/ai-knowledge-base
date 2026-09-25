import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from cache import cache_analysis, cache_clear, cache_list
from graph import ask_graph
from ingest import _watch_loop, ingest_documents, sync_documents


@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=_watch_loop, daemon=True, name="portfolio-watcher").start()
    yield


app = FastAPI(title="AI Knowledge Base - Multi Agent", lifespan=lifespan)


class Question(BaseModel):
    question: str


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
    """Run the compiled LangGraph: capability gate -> local cache -> QA memory -> retrieve -> (direct | LLM)."""
    result = ask_graph.invoke({"question": body.question})
    return {"answer": result["answer"], "sources": result.get("sources", [])}


@app.get("/cache")
def list_cache():
    return cache_list()


@app.get("/cache/analysis")
def analysis_cache():
    return cache_analysis()


@app.delete("/cache")
def clear_cache():
    return cache_clear()