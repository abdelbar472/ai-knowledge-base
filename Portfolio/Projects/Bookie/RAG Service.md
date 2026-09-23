# RAG Service

**Communities 9, 12, 14, 23, 24, 26, 29**

Retrieval-Augmented Generation service using vector search.

### Responsibilities
- Generate embeddings for books
- Index books into vector database (Qdrant)
- Perform semantic search and similarity retrieval

### Key Nodes / Components
**Core:**
- `QdrantClient`
- `RAGEngine`
- `EmbeddingGenerator`

**gRPC & API:**
- `RagService`
- `RagServicer`
- `semantic_search()`
- `find_similar()`
- `sync_books()`
