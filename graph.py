import re
from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph

from cache import local_qa_lookup, local_qa_store
from config import (
    DIRECT_ANSWER_SCORE,
    LEXICAL_BODY_BOOST,
    LEXICAL_META_BOOST,
    LEXICAL_PATH_BOOST,
    RETRIEVE_K,
    SCORE_THRESHOLD,
    TOP_K,
)
from ingest import list_note_files
from store import get_vector_store, search_qa_memory, store_qa_pair

load_dotenv()


# -----------------------------
# LLM
# -----------------------------
llm = ChatGroq(
    model="openai/gpt-oss-20b",   # or llama-3.1-8b-instant if available
    temperature=0
)


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