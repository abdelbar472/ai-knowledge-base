# Auth Service

**Communities 3, 21, 27**

Handles all authentication and authorization logic.

### Responsibilities
- Issue and validate access tokens
- Refresh tokens
- Expose authentication to other microservices via gRPC

### Key Nodes / Components
**Core:**
- `AuthServicer`
- `AuthService`
- `AuthServiceStub`

**Functions:**
- `serve_grpc()`
- `lifespan()`
- Token validation logic

### Related Services

