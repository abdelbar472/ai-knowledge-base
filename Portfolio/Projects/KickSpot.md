---
tags:
  - project
  - uses/microservices
  - uses/grpc
---
# KickSpot

**Two-sided marketplace software for renting football pitches.**

- **Players**: search, book, and play on pitches
- **Owners**: manage pitch listings, availability, and income

---

## Idea Hub

> Brain dump — collect ideas first, refine later.

### The split, restated precisely

Two products under one name:

1. **KickSpot-player** — consumer mobile app for players
2. **KickSpot-owner** — business app for pitch owners/managers

---

### 🎮 KickSpot-player

**Features:**

1. **Auth**
   - [x] splash
   - [x] login
   - [x] register
   - [x] otp verification
   - [x] forget password
   - [x] reset password
   - [x] meta sign in

2. **Home**
   - [ ] discover pitches
   - [ ] pitch details page
   - [ ] check payment
   - [ ] confirm payment
   - [ ] success payment

3. **Booking**
   - [ ] booking page

4. **Friends**
   - [ ] friends page
   - [ ] add friend page
   - [x] invite friend
   - [x] who's playing where page

5. **Profile**
   - [ ] profile page
   - [ ] setting
   - [ ] leaderboard
   - [ ] game history
   - [ ] split cost
   - [ ] notification

### 🏢 KickSpot-owner

- Pitch listings, availability, and income management.

---

## Auth Architecture

### System 1 — Owner/Manager auth (lives inside `owner-service`)

- Two roles: `owner` (full control) and `manager` (scoped to assigned pitch(es))
- Business users — traditional credentials (phone/email + password)
- **RBAC enforcement:** `owner` gets full access; `manager` is scoped via the `PitchManager` join table — sees only assigned pitches, can't touch payouts or add managers
- **Token design:** keep the JWT lean (`user_id` + `role`) and look up pitch assignment from the DB per request — always current, avoids stale-token access bugs

### System 2 — Player auth (lives inside `player-service`)

- WhatsApp-style: **phone number is the identity**, OTP replaces password entirely
- Flow: enter phone → receive SMS OTP → verify → logged in (silent account creation if new)
- No password, ever — long-lived sessions (refresh token stored locally) until explicit logout
- **Flagged cost:** SMS OTPs cost money per message (Twilio, Vonage, or a local Egyptian gateway) — choose a provider with good Egypt coverage when scaling

### Cross-service verification (the careful part)

`booking-service` must verify identity from **either** system — a player creating a booking or a manager marking a no-show. Two issuers, one verifier.

- Both `owner-service` and `player-service` issue JWTs, each signed with their own key
- Every JWT carries an `iss` claim: `"owner-service"` or `"player-service"`
- `booking-service` holds both public keys, checks `iss` first, then verifies with the matching key
- Keeps verification **stateless** — no callback to the issuer on each request
- This belongs in the gRPC contracts when `booking-service` is built

### Open questions

1. Can one person be both an owner and a player? Separate identities vs. shared profile — decide now, hard to merge later.
2. Password reset for owners/managers — email link, or phone+OTP for consistency?
3. JWT signing key storage/rotation — env vars fine for now; production concern later.

---

## Booking Modes

### 1. Private booking

A player/group books a pitch just for themselves, invites known friends or a saved Group.

### 2. Open match / find-opponents

A team of 5 wants to play against another team. Needs:
- A way to **post** "we have 5 players, looking for opponents, at [pitch] on [time]"
- A way for **another team** to **discover and join** that open slot
- Resolution once both sides are filled (confirmed, or first-come-first-served)

### Data model impact (booking-service, later)

New concept beyond `booking_players`:

```
match_requests
├── booking_id          (the pitch slot itself)
├── organizer_id        (player who created it)
├── team_size_needed     (e.g. 5)
├── visibility          (private | open_for_opponents)
├── status              (open | matched | cancelled)
└── opponent_accepted_by (player_id of whoever brought the opposing team, nullable)
```

### Architectural implications

- **H3 indexing** becomes useful for players too — "open matches near me", not just "pitches near me"
- A `booking-service` concern, not `owner-service` — no rework on what's built
- NOT part of Groups/Friends — an opposing team is explicitly *not* your friend group (can optionally intersect)

### Open question

In scope for MVP, or parked with gamification/social features for post-MVP? (Open matches need discovery/browse UI, acceptance flow, notifications.)

---

## Documentation

📁 **Documentation Hub**

- **KickSpot-player**
- **KickSpot-owner**

> Setup instructions, API references, and project structure.