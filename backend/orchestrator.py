import os
import json
import asyncio
import random
import logging
from datetime import date
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage

from llm_provider import get_chat_llm
from .state import AgentState
from .schema import load_metadata, discover_schema_from_db
from .scope_resolver import ScopeResolver, ScopeContext
from .agents import (
    intent_and_planner_agent,
    schema_discovery_agent,
    table_description_agent,
    kpi_fastpath_agent,
    table_selection_agent,
    structure_capture_agent,
    sql_generation_agent,
    review_agent,
    execution_agent,
    relationship_manager_agent,
    natural_language_response_agent,
    compound_query_executor_agent,
    result_merger_agent,
    finalize_response,
    follow_up_agent,
)
from .semantic_cache import semantic_cache


logger = logging.getLogger(__name__)

# Query history is persisted in a dedicated schema in the same database.
HISTORY_SCHEMA = "finfit-ai"
HISTORY_TABLE = f'"{HISTORY_SCHEMA}".query_history'


class SQLAgentOrchestrator:
    def __init__(self, db_config: dict, metadata_dir: str = "./finsight_metadata", model: str | None = None):
        self.db_config = db_config
        self.metadata_dir = metadata_dir
        self.schema_file = os.path.join(metadata_dir, "database_schema.json")
        self.relationships_file = os.path.join(metadata_dir, "table_relationships.json")
        self.table_descriptions_file = os.path.join(metadata_dir, "table_descriptions.json")

        # Provider (OpenAI / Anthropic) and model are chosen from env; `model`
        # is an optional explicit override.
        self.json_llm = get_chat_llm(json_mode=True, temperature=0, model=model)
        self.sql_llm = get_chat_llm(json_mode=False, temperature=0, model=model)
        self.fast_json_llm = get_chat_llm(json_mode=True, temperature=0, model=model)
        self.fast_llm = get_chat_llm(json_mode=False, temperature=0, model=model)
        self.graph = self._build_graph()
        self.cached_schema = None
        self.cached_relationships = None
        self.cached_descriptions = None
        self.pool = None
        # Multi-tenant scope resolution
        self.scope_resolver = None
        self.tenant_scoped_tables = set()
        # Dynamic data window (populated at startup via _load_data_window)
        self.data_min_date: str | None = None
        self.data_max_date: str | None = None

    async def initialize_metadata(self):
        """Full metadata initialization on startup."""
        logger.info("Starting FULL metadata initialization...")
        os.makedirs(self.metadata_dir, exist_ok=True)

        self.cached_schema = load_metadata(self.schema_file)
        self.cached_relationships = load_metadata(self.relationships_file)

        if not (self.cached_schema and self.cached_relationships):
            logger.info("Schema/relationships missing, discovering from DB...")
            try:
                self.cached_schema, self.cached_relationships = await discover_schema_from_db(
                    self.pool, self.schema_file, self.relationships_file
                )
                logger.info("Schema & relationships discovered and cached.")
            except Exception as e:
                raise RuntimeError(f"Failed to connect to database during startup: {e}")
        else:
            logger.info("Schema & relationships loaded from cache.")

        self.cached_descriptions = load_metadata(self.table_descriptions_file)

        if not self.cached_descriptions or len(self.cached_descriptions) == 0:
            logger.info("Table descriptions missing or empty, generating now...")
            dummy_state = AgentState(
                schema_metadata={
                    "tables": self.cached_schema,
                    "relationships": self.cached_relationships or [],
                    "table_descriptions": {}
                },
                messages=[]
            )
            await table_description_agent(self, dummy_state)
            if self.cached_descriptions is None:
                self.cached_descriptions = {}
            logger.info("Generated descriptions for %d tables.", len(self.cached_descriptions))
        else:
            logger.info("Table descriptions loaded from cache (%d tables).", len(self.cached_descriptions))

        logger.info("FULL METADATA READY - All 3 files are complete and up-to-date!")

        await self._load_scope_context()
        await self._load_data_window()
        await self.init_history_store()

    async def _load_scope_context(self):
        """Build the tenant->organization mapping and the set of tenant-scoped tables."""
        # Tables that carry a partner_id column MUST be scope-filtered.
        self.tenant_scoped_tables = set()
        for table_name, details in (self.cached_schema or {}).items():
            columns = details.get("columns", {})
            if "partner_id" in columns:
                short = table_name.split(".")[-1].lower()
                self.tenant_scoped_tables.add(short)
        logger.info("Tenant-scoped tables: %s", sorted(self.tenant_scoped_tables) or '(none)')

        self.scope_resolver = ScopeResolver(self.pool)
        try:
            await self.scope_resolver.load()
        except Exception as e:
            logger.exception("Scope resolver load failed; consolidated requests will be unavailable")

    # All transactional tables that carry a transaction_date column.
    _TRANSACTIONAL_TABLES = (
        "sales_invoice",
        "purchase_invoice",
        "customer_payment",
        "supplier_payment",
        "journal",
    )

    async def _load_data_window(self):
        """Compute real MIN/MAX transaction_date across ALL transactional tables."""
        if not self.pool:
            return

        # Only query tables that actually exist in the discovered schema.
        scoped = self.tenant_scoped_tables or set()
        tables = [t for t in self._TRANSACTIONAL_TABLES if t in scoped]
        if not tables:
            # Fallback: use whatever transactional tables are known
            tables = list(self._TRANSACTIONAL_TABLES)

        union_parts = "\n    UNION ALL\n    ".join(
            f"SELECT transaction_date FROM finance.{t}" for t in tables
        )
        sql = (
            "SELECT MIN(transaction_date) AS mn, MAX(transaction_date) AS mx\n"
            f"FROM (\n    {union_parts}\n) _all_transactions\n"
            "WHERE transaction_date <= CURRENT_DATE"
        )

        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(sql)
            if row and row["mn"] and row["mx"]:
                self.data_min_date = row["mn"].isoformat()
                self.data_max_date = row["mx"].isoformat()
                logger.info(
                    "Data window (across %s): %s → %s",
                    ", ".join(f"finance.{t}" for t in tables),
                    self.data_min_date,
                    self.data_max_date,
                )
            else:
                logger.warning("Data window query returned no rows; dates remain unset.")
        except Exception as e:
            logger.warning("Could not compute data window: %s", e)

    async def get_data_window(self) -> tuple[str, str]:
        """Return (min_date, max_date) fetched from the DB; re-queries if not yet cached."""
        if self.data_min_date is None or self.data_max_date is None:
            await self._load_data_window()
        # Ultimate fallback & future-date safety guard
        today_iso = date.today().isoformat()
        min_d = self.data_min_date or today_iso
        max_d = min(self.data_max_date, today_iso) if self.data_max_date else today_iso
        return (min_d, max_d)

    # --- Query history store (persisted in the "finfit-ai" schema) ---

    async def init_history_store(self):
        """Create the finfit-ai schema and query_history table if they do not exist."""
        if not self.pool:
            return
        try:
            async with self.pool.acquire() as conn:
                await conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{HISTORY_SCHEMA}"')
                await conn.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {HISTORY_TABLE} (
                        id BIGSERIAL PRIMARY KEY,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                        scope_mode TEXT,
                        tenant_id INTEGER,
                        allowed_partner_ids INTEGER[],
                        user_query TEXT NOT NULL,
                        sql_query TEXT,
                        row_count INTEGER,
                        success BOOLEAN NOT NULL DEFAULT TRUE
                    )
                    """
                )
            logger.info("Query history store ready at %s", HISTORY_TABLE)
        except Exception as e:
            logger.warning("Failed to initialize query history store: %s", e)

    async def save_query_history(self, entry: dict):
        """Insert a single query-history record. Failures are logged, never raised."""
        if not self.pool:
            return
        try:
            async with self.pool.acquire() as conn:
                await conn.execute(
                    f"""
                    INSERT INTO {HISTORY_TABLE}
                        (scope_mode, tenant_id, allowed_partner_ids,
                         user_query, sql_query, row_count, success)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    """,
                    entry.get("scope_mode"),
                    entry.get("tenant_id"),
                    entry.get("allowed_partner_ids"),
                    entry.get("user_query"),
                    entry.get("sql_query"),
                    entry.get("row_count"),
                    entry.get("success", True),
                )
        except Exception as e:
            logger.warning("Failed to save query history: %s", e)

    async def get_query_history(self, limit: int = 10) -> list[dict]:
        """Return the most recent `limit` records in chronological order (oldest first)."""
        if not self.pool:
            return []
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT created_at, scope_mode, tenant_id, allowed_partner_ids,
                       user_query, sql_query, row_count, success
                FROM {HISTORY_TABLE}
                ORDER BY id DESC
                LIMIT $1
                """,
                limit,
            )
        history = [
            {
                "timestamp": r["created_at"].isoformat() if r["created_at"] else None,
                "scope_mode": r["scope_mode"],
                "tenant_id": r["tenant_id"],
                "allowed_partner_ids": list(r["allowed_partner_ids"]) if r["allowed_partner_ids"] else None,
                "user_query": r["user_query"],
                "sql_query": r["sql_query"],
                "row_count": r["row_count"],
                "success": r["success"],
            }
            for r in rows
        ]
        history.reverse()
        return history

    async def count_query_history(self) -> int:
        """Total number of stored query-history records."""
        if not self.pool:
            return 0
        async with self.pool.acquire() as conn:
            return await conn.fetchval(f"SELECT COUNT(*) FROM {HISTORY_TABLE}") or 0

    async def clear_query_history(self):
        """Delete all stored query-history records."""
        if not self.pool:
            return
        async with self.pool.acquire() as conn:
            await conn.execute(f"TRUNCATE {HISTORY_TABLE} RESTART IDENTITY")

    def _build_graph(self):
        """Build the LangGraph workflow."""
        orch = self
        workflow = StateGraph(AgentState)

        async def _intent_and_planner(state): return await intent_and_planner_agent(orch, state)
        async def _schema_discovery(state): return await schema_discovery_agent(orch, state)
        async def _table_description(state): return await table_description_agent(orch, state)
        async def _kpi_fastpath(state): return await kpi_fastpath_agent(orch, state)
        async def _table_selection(state): return await table_selection_agent(orch, state)
        async def _structure_capture(state): return await structure_capture_agent(orch, state)
        async def _sql_generation(state): return await sql_generation_agent(orch, state)
        async def _review(state): return await review_agent(orch, state)
        async def _execution(state): return await execution_agent(orch, state)
        async def _relationship_manager(state): return await relationship_manager_agent(orch, state)
        async def _nl_response(state): return await natural_language_response_agent(orch, state)
        async def _compound_executor(state): return await compound_query_executor_agent(orch, state)
        async def _result_merger(state): return await result_merger_agent(orch, state)
        async def _follow_up(state): return await follow_up_agent(orch, state)
        def _finalize(state): return finalize_response(orch, state)

        workflow.add_node("intent_and_planner", _intent_and_planner)
        workflow.add_node("schema_discovery", _schema_discovery)
        workflow.add_node("table_description", _table_description)
        workflow.add_node("kpi_fastpath", _kpi_fastpath)
        workflow.add_node("table_selection", _table_selection)
        workflow.add_node("structure_capture", _structure_capture)
        workflow.add_node("sql_generation", _sql_generation)
        workflow.add_node("review", _review)
        workflow.add_node("execution", _execution)
        workflow.add_node("relationship_manager", _relationship_manager)
        workflow.add_node("natural_language_response", _nl_response)
        workflow.add_node("compound_executor", _compound_executor)
        workflow.add_node("result_merger", _result_merger)
        workflow.add_node("follow_up", _follow_up)
        workflow.add_node("finalize", _finalize)

        # Entry: single LLM call classifies intent AND plans decomposition
        workflow.set_entry_point("intent_and_planner")
        workflow.add_conditional_edges("intent_and_planner", self._check_intent, {
            "proceed": "schema_discovery",
            "abort":   "finalize",
            "clarify": "finalize",
            "cache_hit": "review",
        })
        workflow.add_edge("schema_discovery", "table_description")

        # After table descriptions are ready, branch: compound vs simple
        workflow.add_conditional_edges("table_description", self._route_query, {
            "compound": "compound_executor",
            "simple":   "kpi_fastpath",
        })

        # Compound path: executor → merger → follow_up → finalize
        workflow.add_edge("compound_executor", "result_merger")
        workflow.add_edge("result_merger", "follow_up")

        # Simple path: existing KPI fast-path → full SQL pipeline
        workflow.add_conditional_edges("kpi_fastpath", self._kpi_fastpath_router, {
            "hit":  "natural_language_response",
            "miss": "table_selection",
        })
        workflow.add_edge("table_selection", "structure_capture")
        workflow.add_edge("structure_capture", "sql_generation")
        workflow.add_edge("sql_generation", "review")
        workflow.add_edge("review", "execution")
        workflow.add_edge("natural_language_response", "follow_up")
        workflow.add_edge("follow_up", "finalize")

        workflow.add_conditional_edges("execution", self._should_retry, {
            "retry": "sql_generation",
            "expand_relationships": "relationship_manager",
            "finish": "natural_language_response"
        })

        workflow.add_conditional_edges("relationship_manager", self._should_expand, {
            "regenerate": "structure_capture",
            "finish": "natural_language_response"
        })

        workflow.add_edge("finalize", END)
        return workflow.compile()

    def _check_intent(self, state: AgentState) -> str:
        """Route to schema_discovery (proceed) or finalize (abort/clarify) based on intent."""
        if state.get("generated_sql"):
            return "cache_hit"
        if state.get("clarification_needed"):
            return "clarify"
        intent = state.get("intent", "finance_query")
        if intent in ("off_topic", "ambiguous"):
            return "abort"
        return "proceed"

    def _route_query(self, state: AgentState) -> str:
        """Route to compound_executor or kpi_fastpath based on planner output."""
        return "compound" if state.get("is_compound") and state.get("sub_query_plan") else "simple"

    def _kpi_fastpath_router(self, state: AgentState) -> str:
        """Route to natural_language_response (hit) or table_selection (miss)."""
        return "hit" if state.get("kpi_fastpath_hit") else "miss"

    async def _should_retry(self, state: AgentState) -> str:
        """Decide if query should be retried. retry_count is incremented inside nodes."""
        MAX_RETRIES = 4
        retry_count = state.get("retry_count", 0)
        exec_result = state.get("execution_result", {})

        if state.get("error") and retry_count < MAX_RETRIES:
            delay = min(1.5 ** retry_count + random.uniform(0, 1), 10)
            logger.warning(
                "Retrying SQL generation: attempt=%d/%d delay=%.1fs reason=%s",
                retry_count + 1, MAX_RETRIES, delay, str(state.get("error"))[:200],
            )
            await asyncio.sleep(delay)
            return "retry"

        if not exec_result.get("success") and retry_count < MAX_RETRIES:
            delay = min(1.5 ** retry_count + random.uniform(0, 1), 10)
            logger.warning(
                "Retrying SQL generation: attempt=%d/%d delay=%.1fs reason=execution_failed error=%s",
                retry_count + 1, MAX_RETRIES, delay, exec_result.get("error", ""),
            )
            await asyncio.sleep(delay)
            return "retry"

        if exec_result.get("success") and exec_result.get("row_count", 0) == 0 and retry_count < 1:
            logger.info("No rows returned — expanding table relationships for broader query coverage")
            return "expand_relationships"
        return "finish"

    def _should_expand(self, state: AgentState) -> str:
        """Decide if relationships should be expanded."""
        if state.get("related_tables") and state.get("retry_count", 0) < 1:
            return "regenerate"
        return "finish"

    async def process_query(self, user_query: str, scope: ScopeContext, history: list = None):
        """Main entry point: run the full agent workflow for a resolved scope."""
        logger.info(
            "Query started: query=%r scope_mode=%s tenant_id=%r partner_ids=%s",
            user_query, scope.mode, scope.tenant_id, scope.allowed_partner_ids,
        )

        if not scope.allowed_partner_ids:
            raise ValueError("No allowed partner_ids resolved for this request.")

        initial_state = {
            "messages": [HumanMessage(content=user_query)],
            "user_query": user_query,
            "db_config": self.db_config,
            "schema_metadata": {},
            "selected_tables": [],
            "table_structures": {},
            "generated_sql": "",
            "validation_result": {},
            "execution_result": {},
            "retry_count": 0,
            "related_tables": [],
            "final_response": "",
            "error": "",
            "scope_mode": scope.mode,
            "tenant_id": scope.tenant_id,
            "allowed_partner_ids": list(scope.allowed_partner_ids),
            "org_names": dict(scope.org_names),
            "natural_language_answer": "",
            "intent": "",
            "kpi_fastpath_hit": False,
            "history": history or [],
            "clarification_needed": False,
            "clarification_question": "",
            "is_compound": False,
            "sub_query_plan": [],
            "sub_query_results": [],
        }

        config = {"recursion_limit": 50}

        try:
            final_state = await self.graph.ainvoke(initial_state, config=config)
            logger.info(
                "Query completed: query=%r scope_mode=%s tenant_id=%r "
                "retry_count=%d success=%s",
                user_query,
                scope.mode,
                scope.tenant_id,
                final_state.get("retry_count", 0),
                final_state.get("execution_result", {}).get("success", False),
            )
            raw = final_state.get("final_response") or ""
            if raw:
                return json.loads(raw)
            # Graph completed but produced no response body: fall through to the
            # honest structured error below rather than raising an opaque 500.
            return self._error_response(
                state_error=final_state.get("error", ""),
                generated_sql=final_state.get("generated_sql", ""),
            )
        except Exception as e:
            logger.exception(
                "Query processing failed: query=%r scope_mode=%s tenant_id=%r",
                user_query, scope.mode, scope.tenant_id,
            )
            # Return a structured, honest response instead of raising an opaque
            # 500. The most common real-world cause is an invalid / expired
            # OPENAI_API_KEY (the SQL-generation LLM is mandatory for this
            # feature and cannot be templated), so surface that clearly.
            return self._error_response(state_error=str(e), generated_sql="")

    async def process_query_stream(self, user_query: str, scope: ScopeContext, history: list = None):
        """Execute the query pipeline and yield SSE events (steps and tokens)."""
        if not scope.allowed_partner_ids:
            yield f"data: {json.dumps({'status': 'error', 'answer': 'No allowed partner_ids resolved for this request.'})}\n\n"
            return

        cached_sql = await semantic_cache.search(user_query, scope.tenant_id, list(scope.allowed_partner_ids))

        initial_state = {
            "messages": [HumanMessage(content=user_query)],
            "user_query": user_query,
            "db_config": self.db_config,
            "schema_metadata": {},
            "selected_tables": [],
            "table_structures": {},
            "generated_sql": cached_sql or "",
            "validation_result": {},
            "execution_result": {},
            "retry_count": 0,
            "related_tables": [],
            "final_response": "",
            "error": "",
            "scope_mode": scope.mode,
            "tenant_id": scope.tenant_id,
            "allowed_partner_ids": list(scope.allowed_partner_ids),
            "org_names": dict(scope.org_names),
            "natural_language_answer": "",
            "intent": "",
            "kpi_fastpath_hit": False,
            "history": history or [],
            "clarification_needed": False,
            "clarification_question": "",
            "is_compound": False,
            "sub_query_plan": [],
            "sub_query_results": [],
        }

        config = {"recursion_limit": 50}

        try:
            # Yield initial step
            yield f"data: {json.dumps({'type': 'step', 'step': 'Initializing Pipeline'})}\n\n"

            final_state = None
            async for event in self.graph.astream_events(initial_state, config=config, version="v2"):
                kind = event["event"]
                tags = event.get("tags", [])
                
                if kind == "on_chain_end":
                    name = event.get("name")
                    if name in self.graph.nodes:
                        step_name = name.replace("_", " ").title()
                        yield f"data: {json.dumps({'type': 'step', 'step': step_name})}\n\n"
                    # Capture the final state from the root graph
                    if name == "LangGraph":
                        final_state = event.get("data", {}).get("output", {})
                
                elif kind == "on_chat_model_stream" and "final_answer_llm" in tags:
                    chunk = event["data"]["chunk"].content
                    if chunk:
                        yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"
                        
            # Graph finished
            if not final_state:
                final_state = {}
                
            raw = final_state.get("final_response") or ""
            if raw:
                # Save successful runs to cache
                if not cached_sql and final_state.get("execution_result", {}).get("success") and not final_state.get("error"):
                    gen_sql = final_state.get("generated_sql")
                    if gen_sql:
                        await semantic_cache.add(user_query, gen_sql, scope.tenant_id, list(scope.allowed_partner_ids))
                        
                final_dict = json.loads(raw)
                final_dict["type"] = "final"
                yield f"data: {json.dumps(final_dict)}\n\n"
            else:
                err = self._error_response(state_error=final_state.get("error", ""), generated_sql=final_state.get("generated_sql", ""))
                err["type"] = "final"
                yield f"data: {json.dumps(err)}\n\n"

        except Exception as e:
            logger.exception("Stream processing failed: %s", e)
            err = self._error_response(state_error=str(e), generated_sql="")
            err["type"] = "final"
            yield f"data: {json.dumps(err)}\n\n"

    @staticmethod
    def _error_response(*, state_error: str, generated_sql: str) -> dict:
        """Build a graceful, user-facing error payload for the chatbot."""
        blob = (state_error or "").lower()
        is_auth = any(
            token in blob
            for token in ("401", "invalid_api_key", "incorrect api key", "api key", "authentication")
        )
        if is_auth:
            answer = (
                "The AI assistant is temporarily unavailable because its language-model "
                "credential is not valid. Please contact your administrator to refresh the "
                "AI service API key."
            )
            error = "LLM authentication failed (invalid or expired API key)."
        elif not generated_sql:
            answer = (
                "I couldn't turn that question into a database query. Please rephrase it, or "
                "try a more specific question about your finance data."
            )
            error = state_error or "No SQL was generated for the query."
        else:
            answer = "Sorry, I couldn't complete that request. Please try again."
            error = state_error or "Query processing failed."
        return {
            "status": "error",
            "answer": answer,
            "error": error,
            "sql": generated_sql,
            "row_count": 0,
            "steps": [],
        }
