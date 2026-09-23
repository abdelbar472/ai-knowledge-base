# User Service

**Community 0** • Cohesion: **0.05** (Low) • 81 nodes

Core service for managing users and their profiles.

### Responsibilities
- User registration and profile CRUD operations
- Profile updates and token refresh
- Database migrations for user data

### Key Nodes / Components
**Models & Requests:**
- `ProfileUpdate`
- `RefreshTokenResponse`
- `TokenRefreshRequest`

**Database & Utils:**
- `create_db_and_tables()`
- `get_session()`
- `_migrate_user_profiles_table()`

**gRPC:**
- `User Authserviceservicer`

### Related Services
