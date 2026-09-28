# SQL Agent — Workflow & Architecture

The SQL Agent converts natural-language questions into safe, **tenant-scoped** SQL queries against a PostgreSQL finance database. It runs a multi-node **LangGraph** state machine with deterministic scope enforcement, automatic retry, and relationship expansion. Every request emits a structured **step-by-step trace** so the workflow is fully observable in the UI.

---

## High-Level Flow

```
POST /api/v1/query
  headers: consolidated-view, tenant-id, organization-id
        │
        ▼
┌───────────────────────┐
│  Scope Resolution     │  ScopeResolver.resolve(...)  → ScopeContext
│  (single | consolidated)│  (runs BEFORE the graph)
└───────────┬───────────┘
            ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        LangGraph State Machine                         │
│                                                                        │
│  ┌────────────┐   ┌──────────────┐   ┌──────────────┐                 │
│  │ 1. Schema  │──▶│ 2. Table     │──▶│ 3. Table     │                 │
│  │  Discovery │   │  Description │   │  Selection   │                 │
│  └────────────┘   └──────────────┘   └──────┬───────┘                 │
│                                             ▼                          │
│  ┌────────────┐   ┌──────────────┐   ┌──────────────┐                 │
│  │ 6. SQL     │◀──│ 5. SQL       │◀──│ 4. Structure │                 │
│  │  Review    │   │  Generation  │   │  Capture     │                 │
│  └─────┬──────┘   └──────────────┘   └──────────────┘                 │
│        ▼                                                               │
│  ┌────────────┐                                                       │
│  │ 7. Execution│  ── scope guard (sqlglot AST) enforced here          │
│  └─────┬──────┘                                                       │
│        │                                                              │
│        ├── retry ─────────▶ back to SQL Generation (error context)    │
│        ├── expand ───────▶ Relationship Manager ─▶ Structure Capture  │
│        │                                                              │
│  ┌─────▼──────┐   ┌──────────────┐                                    │
│  │ 8. Natural │──▶│ 9. Finalize  │──▶ Response (answer, sql,          │
│  │  Language  │   │  Response    │      row_count, steps[])           │
│  └────────────┘   └──────────────┘                                    │
└──────────────────────────────────────────────────────────────────────┘
```

---

## API Contract

### Request

Scope is supplied via **HTTP headers** (primary transport; body fields are accepted as a fallback).

| Header | Type | Meaning |
|--------|------|---------|
| `consolidated-view` | `true` / `false` | If true, aggregate across **all** organizations in a tenant |
| `tenant-id` | integer | Required when `consolidated-view=true` |
| `organization-id` | integer | Required for single-organization queries (the `partner_id`) |

> **Note:** Scope headers use **hyphenated** names (`consolidated-view` / `tenant-id` / `organization-id`) because reverse proxies (nginx/ingress) strip headers containing underscores by default. Hyphenated headers pass through every proxy unchanged.

Body:

```json
{ "query": "monthly journal entry count for this year" }
```

Fallback body fields (used only when the matching header is absent): `consolidatedView`, `tenantId`, `organizationId`, `partnerId`.

### Response

```json
{
  "status": "success",
  "answer": "For BONBLOC TECHNOLOGIES LIMITED in 2025, the monthly journal entry counts are ...",
  "sql": "SELECT DATE_TRUNC('month', j.transaction_date) ...",
  "row_count": 12,
  "processing_time": 8.55,
  "steps": [
    { "step": "Schema Discovery",           "status": "done",  "detail": "Loaded database schema (6 tables available)." },
    { "step": "Scope Resolution",           "status": "done",  "detail": "Single organization — partner_id 91." },
    { "step": "Table Selection",            "status": "done",  "detail": "finance.journal" },
    { "step": "SQL Generation",             "status": "done",  "detail": "SELECT ...", "code": true },
    { "step": "Review & Validation",        "status": "done",  "detail": "Query validated — no issues." },
    { "step": "Execution",                  "status": "done",  "detail": "Query executed successfully — 12 row(s) returned." },
    { "step": "Natural Language Response",  "status": "done",  "detail": "Formatted the result into a human-readable answer." }
  ]
}
```

On error, `status` is `"error"`, `error` carries the message, and `steps[]` still describes how far the pipeline got (the failing step has `status: "error"`).

---

## Scope Resolution (before the graph)

`ScopeResolver` (`sql_agent/scope_resolver.py`) builds a tenant → organization map at startup from `public.organization` (`organization.id` == `finance.partner_id`, `tenant_id` non-null). `resolve(...)` produces a deterministic `ScopeContext`:

| Mode | Trigger | `allowed_partner_ids` |
|------|---------|------------------------|
| **single** | `organization_id` provided | `[organization_id]` |
| **consolidated** | `consolidated_view=true` + `tenant_id` | every `partner_id` mapped to that tenant |

```python
ScopeContext(
    mode="consolidated",
    tenant_id=8,
    allowed_partner_ids=[88, 91, 92],
    org_names={88: "BONbLOC Technologies", 91: "BONBLOC TECHNOLOGIES LIMITED", 92: "BONbLOC USA"},
)
```

Invalid requests (missing `tenant_id` in consolidated mode, unknown tenant, missing `organization_id` in single mode) raise `ScopeError` → HTTP `400`. The resolved scope is injected into `AgentState` (`scope_mode`, `tenant_id`, `allowed_partner_ids`, `org_names`).

---

## Stage 1 — Schema Discovery (`schema.py`)

Loads the database structure (tables, columns, types, PKs, FKs). Three-tier caching:

1. **In-memory** — `orchestrator.cached_schema`
2. **Disk** — `finsight_metadata/database_schema.json`
3. **Live discovery** — `information_schema` + `pg_catalog` (only if cache missing or `POST /refresh-schema`)

At startup the orchestrator also:
- Derives `tenant_scoped_tables` = every table that has a `partner_id` column (these **must** be scope-filtered).
- Computes a **dynamic data window** (`MIN`/`MAX transaction_date`) from `sales_invoice` / `purchase_invoice` / `journal`, falling back to `2025-04-01 → 2026-05-31`.

---

## Stage 2 — Table Description (`agents.py → table_description_agent`)

Generates a 1–2 sentence AI description per table so the selection LLM has semantic context. Cached to `table_descriptions.json`. Runs once at startup (or on `/refresh-schema`).

---

## Stage 3 — Table Selection (`agents.py → table_selection_agent`)

The LLM picks which tables answer the question, using table names + descriptions. Fallbacks: keyword match against table names → first few tables.

```json
{ "selected_tables": ["finance.journal"], "reasoning": "The query asks about journal entry counts." }
```

---

## Stage 4 — Structure Capture (`agents.py → structure_capture_agent`)

Pulls the full column/FK detail for **only the selected tables**, keeping the SQL-generation prompt small and focused.

---

## Stage 5 — SQL Generation (`agents.py → sql_generation_agent`)

Generates scope-aware PostgreSQL. The prompt injects:

| Section | Content |
|---------|---------|
| **Scope rule** | single → `partner_id = <id>`; consolidated → `partner_id IN (<ids>)` |
| **Date rule** | filter on `transaction_date` (never `posting_period`) |
| **Data window** | the dynamic `MIN → MAX` range; no `CURRENT_DATE` / `NOW()` |
| **Schema context** | exact columns/types for selected tables |
| **Relationships** | FKs between selected tables |
| **Retry context** | previous SQL + error message when regenerating |

For consolidated queries the partner filter is built by `_partner_filter_sql(...)` as an `IN (...)` list, and grouping by organization is added when the answer benefits from a per-org breakdown.

---

## Stage 6 — SQL Review (`agents.py → review_agent`)

A second LLM pass that validates before execution:
1. Non-existent columns / wrong table references
2. Syntax errors and missing JOIN conditions
3. Missing / wrong `partner_id` filter
4. Use of `posting_period` for date filtering (wrong → must use `transaction_date`)
5. Use of `CURRENT_DATE` / `NOW()` (wrong → must hard-code the window)

```json
{ "valid": false,
  "issues": ["Query uses a date range beyond the available data window."],
  "corrected_sql": "SELECT ... WHERE transaction_date >= '2025-04-01' AND transaction_date < '2026-06-01'" }
```

The corrected SQL replaces the original before execution. (This is the same "Issues found and corrected" text surfaced in the UI trace.)

---

## Stage 7 — Execution (`agents.py → execution_agent`)

Two independent safety layers run **before** the query touches the database:

**1. Static safety** — must start with `SELECT`/`WITH`, blocklist of dangerous keywords, 30 s timeout, read-only transaction:

```python
DANGEROUS_KEYWORDS = ['DROP','DELETE','TRUNCATE','ALTER','CREATE','INSERT','UPDATE',
                      'GRANT','REVOKE','EXEC','EXECUTE','CALL','SET','LOCK','UNLOCK']
```

**2. Scope guard** — `enforce_partner_scope(sql, allowed_partner_ids, tenant_scoped_tables)` (`sql_agent/scope_guard.py`). This is the **real security boundary**. Using `sqlglot`'s AST (not regex) it proves:

- Every tenant-scoped table referenced is filtered on `partner_id`.
- `partner_id` is only used with `=` or `IN (<int list>)` — never `!=`, `>`, `<`, `NOT IN`, or an `IN (subquery)`.
- Every `partner_id` literal is an integer **inside** the caller's `allowed_partner_ids`.

If the query cannot be proven safe, execution is refused (`ScopeViolation`) regardless of what the LLM produced. CTEs, sub-queries, aliases, and joins are all handled via the parse tree.

Successful results are recorded in the `"finfit-ai".query_history` table (same database).

---

## Conditional Routing — Retry & Expand

After execution the graph decides the next edge (`orchestrator._should_retry`):

```
├─ error / failure  &  retry_count < 4   → "retry"
│    back to SQL Generation with error context
│    exponential backoff: min(1.5^n + jitter, 10s)
│
├─ success, 0 rows,  never expanded      → "expand_relationships"
│    Relationship Manager finds FK-linked tables, loops to Structure Capture
│
└─ success with rows  OR  retries exhausted → "finish" → Natural Language Response
```

`MAX_RETRIES = 4`; relationship expansion happens at most once.

---

## Relationship Manager (`agents.py → relationship_manager_agent`)

When a query returns 0 rows, it inspects foreign keys to find related tables that might hold the data (max 3), then the workflow loops back through Structure Capture → SQL Generation with the expanded table set.

---

## Stage 8 — Natural Language Response (`agents.py → natural_language_response_agent`)

Converts raw rows into a readable answer. For consolidated results it formats per-organization breakdowns and totals. For empty results it explains the available data window and suggests rephrasing.

---

## Stage 9 — Finalize Response (`agents.py → finalize_response` + `_build_steps`)

Packages the API payload and builds the observability trace:

- `finalize_response` assembles `status`, `answer`, `sql`, `row_count`, and `steps`.
- `_build_steps(state)` derives the ordered `steps[]` array (Schema Discovery → Scope Resolution → Table Selection → SQL Generation `{code:true}` → Review & Validation → Execution → optional Retries → Natural Language Response), each with `status` (`done` / `error` / `skipped` / `info`) and a human-readable `detail`.

---

## Security & Safety Summary

| Protection | Implementation |
|------------|----------------|
| Read-only enforcement | Must start with `SELECT`/`WITH` + dangerous-keyword blocklist |
| Tenant isolation (hard) | `enforce_partner_scope` AST proof against `allowed_partner_ids` |
| Scope resolution | `ScopeResolver` maps tenant ↔ organizations from `public.organization` |
| Deterministic data window | Real `MIN`/`MAX transaction_date`, no `CURRENT_DATE`/`NOW()` |
| SQL injection prevention | LLM generates SQL; no user-string interpolation into queries |
| Execution timeout | 30 s per query |
| Retry limit | Max 4 (prevents infinite loops / token burn) |
| Exponential backoff | `min(1.5^n + jitter, 10s)` |

---

## Metadata Caching

Stored under `finsight_metadata/`:

| File | Contents | Rebuilt on |
|------|----------|-----------|
| `database_schema.json` | Tables, columns, types, PKs, FKs | Startup (if missing) or `/refresh-schema` |
| `table_relationships.json` | FK relationships between tables | Same as above |
| `table_descriptions.json` | AI-generated table descriptions | Same as above |
| `"finfit-ai".query_history` (DB table) | Query + SQL + scope + row count | Every successful execution |

The full metadata (schema, descriptions, tenant scope, data window) is also exposed read-only via `GET /api/v1/metadata` and rendered in the frontend **Metadata** tab.

---

## File Reference

| File | Purpose |
|------|---------|
| `sql_agent/state.py` | `AgentState` TypedDict (adds `scope_mode`, `tenant_id`, `allowed_partner_ids`, `org_names`) |
| `sql_agent/schema.py` | DB discovery, metadata I/O, static SQL safety |
| `sql_agent/scope_resolver.py` | Tenant ↔ organization mapping, `ScopeContext` |
| `sql_agent/scope_guard.py` | AST-based `enforce_partner_scope` (hard tenant isolation) |
| `sql_agent/agents.py` | All LangGraph node functions + `_build_steps` / `finalize_response` |
| `sql_agent/orchestrator.py` | Init, graph wiring, retry logic, `process_query` |
| `app.py` | FastAPI endpoints (`/api/v1/query`, `/api/v1/metadata`, `/refresh-schema`, …) |

---

## Scope Examples

### Single organization
```
headers: organization-id: 91
body:    { "query": "total sales invoice amount this month" }
→ SELECT SUM(grand_total) FROM finance.sales_invoice
  WHERE partner_id = 91 AND transaction_date >= ... AND transaction_date < ...
```

### Consolidated (tenant-wide)
```
headers: consolidated-view: true, tenant-id: 8
body:    { "query": "total purchase invoice amount broken down by organization" }
→ SELECT partner_id, SUM(grand_total) FROM finance.purchase_invoice
  WHERE partner_id IN (88, 91, 92) GROUP BY partner_id
```

### Blocked by scope guard (cross-tenant attempt)
```
headers: organization-id: 91
generated: ... WHERE partner_id = 26   -- outside allowed scope
→ refused: "Query references partner_id(s) [26] outside the allowed scope [91]."
```
