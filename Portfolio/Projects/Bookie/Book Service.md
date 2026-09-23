# Book Service

**Communities 11, 25, 33, 35**

Central service for book metadata and enrichment.

### Responsibilities
- Retrieve books by ISBN
- Auto-enrichment of book data
- Provide internal gRPC API for other services

### Key Nodes / Components
**gRPC:**
- `BookService`
- `BookV3Service`
- `BookServiceServicer`
- `BookServiceStub`

**Features:**
- Get book with auto enrichment
- Backward compatibility support
