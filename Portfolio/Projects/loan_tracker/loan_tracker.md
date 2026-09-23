---
tags:
  - project
  - uses/go
  - uses/sqlite
  - uses/nextjs
  - uses/react
  - uses/tailwind
  - uses/rest
  - uses/typescript
---
# Personal Loan & Asset Tracking Application

[](https://github.com/abdelbar472/loan_tracker#personal-loan--asset-tracking-application)

A personal-finance web app for tracking loans you've **lent** or **borrowed** (in any currency or gold), bank accounts, and a death-time "successor" handover. Built as an **independent Go JSON API** (frontend-agnostic) with a **Next.js** reference frontend.

- **Backend:** Go (module `loan`) — plain JSON REST API over SQLite.
- **Frontend:** Next.js (App Router) reference app in `frontend/`.
- **Money:** always integer minor-units — **float64 is never used for amounts**.

---

## Features

[](https://github.com/abdelbar472/loan_tracker#features)

- **Loans** — track money _out_ (lent) and money _in_ (borrowed); multi-unit amounts (`LOCAL`, `USD`, `GOLD_24K`).
- **Rate stamping** — each loan/payment keeps the FX gold rate locked at creation/payment time; historical values are never recomputed from a live rate.
- **Netting / "make it even"** — when you've both lent to and borrowed from the _same person_, counterparties are grouped by identity and the dashboard shows a single **net** amount (and netted totals) instead of two opposing line items. Matching mutual loans **auto-settle**: equal lent↔borrowed pairs resolve to `collected`/`settled` in the database, and unequal pairs are reduced to the net remainder (`partially_paid`). Pre-existing offsetting loans are reconciled on startup and on every read.
- **Payments** — partial payments drive status: `active` → `partially_paid` → `settled` / `collected`; loans can also be `written_off`.
- **Bank accounts** — masked account numbers (`****` + last 4) with balances.
- **Successors & death handover** — designate a successor; on confirmed death the deceased account is **frozen** (login and all operations rejected) and the successor can resolve the estate: view the deceased user's loans, record payments on their behalf, and mark each loan as **checked off**. The death-time portfolio snapshot is generated as a PDF artifact.
- **Deceased accounts cannot take on debt** — creating a loan whose counterparty is a deceased registered user is rejected in full, so no new borrowed (or lent) mirror can be recorded against the estate.
- **Inactivity escalation** — a separate worker flags inactive accounts, warns the owner, then escalates to `pending_verification` and notifies successors.
- **Notifications** — per-user in-app feed with unread counts.

---

## Tech stack

[](https://github.com/abdelbar472/loan_tracker#tech-stack)

|Layer|Technology|
|---|---|
|Backend|Go 1.26, stdlib `net/http` only (no web framework)|
|Database|SQLite via `modernc.org/sqlite` (pure Go, no CGO)|
|Auth|JWT (`github.com/golang-jwt/jwt/v5`), bcrypt passwords, bearer token|
|Money|`platform/money` — integer minor-units|
|Rates|`platform/fxrate` — TTL-cached provider (stub)|
|Frontend|Next.js 16 + React 19 + Tailwind CSS 4|