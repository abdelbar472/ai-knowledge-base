from pathlib import Path

# -----------------------------
# Config
# -----------------------------
PORTFOLIO_PATH = "./Portfolio"          # your notes
COLLECTION_NAME = "knowledge_base"
QDRANT_URL = "http://localhost:6333"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_SIZE = 384

# Retrieval / answering
TOP_K = 4
SCORE_THRESHOLD = 0.5      # dense gate for the candidate POOL; final order is the hybrid rerank
RETRIEVE_K = 24             # candidate pool before the hybrid lexical rerank (TOP_K survive)
LEXICAL_BODY_BOOST = 0.03    # relevance added per question token found in the chunk text
LEXICAL_PATH_BOOST = 0.09    # relevance added per question token found in the source path (folder/file = high signal)
LEXICAL_META_BOOST = 0.05    # relevance added per question token found in frontmatter (category/summary/tech)
DIRECT_ANSWER_SCORE = 0.90   # top chunk this confident -> answer directly from the KB, no LLM
QA_HIT_SCORE = 0.82          # a stored Q->A pair this confident -> reuse it, no LLM

# QA memory (Qdrant vector collection) + pipeline versioning
QA_COLLECTION = "qa_memory"  # keeps generated Q->A pairs separate from the raw notes
PIPELINE_VERSION = "rag-hybrid-v1"  # bumped when retrieval/LLM behaviour changes -> stale cached answers are ignored

# Storage locations (all runtime-generated, git-ignored)
MANIFEST_PATH = Path(".ingest_manifest.json")
EMBED_CACHE_DIR = Path(".vector_cache")  # stored embeddings, reused before re-embedding
LOCAL_QA_DB = Path("_local_qa_cache.sqlite3")