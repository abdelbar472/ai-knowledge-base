# Follow Service

**Communities 2, 20, 22, 28**

Manages social following relationships between users.

### Responsibilities
- Follow / unfollow users
- Retrieve followers and following lists
- Get follow statistics

### Key Nodes / Components
**Functions:**
- `get_followers()`
- `get_following()`
- `is_following()`
- `get_follow_stats()`

**gRPC:**
- `FollowService`
- `FollowServiceStub`
- `FollowServiceServicer`
