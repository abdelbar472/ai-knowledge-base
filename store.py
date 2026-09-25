import uuid

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams
from langchain_qdrant import QdrantVectorStore

from config import (
    COLLECTION_NAME,
    EMBEDDING_SIZE,
    PIPELINE_VERSION,
    QA_COLLECTION,
    QA_HIT_SCORE,
    QDRANT_URL,
)
from embeddings import get_embeddings

client = QdrantClient(url=QDRANT_URL)


def get_vector_store() -> QdrantVectorStore:
    return QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=get_embeddings(),
        distance=Distance.COSINE,
    )


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