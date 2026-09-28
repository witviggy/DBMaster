import json
import logging
import re
import asyncio
from datetime import datetime, date
from langchain_core.messages import HumanMessage, AIMessage

from .state import AgentState
from .schema import load_metadata, save_metadata, validate_sql_safety
from .scope_guard import enforce_partner_scope
from .date_utils import apply_relative_date_filter, derive_relative_date_filter, resolve_posting_period_hint

logger = logging.getLogger(__name__)


def _partner_filter_sql(allowed_partner_ids: list) -> str:
    """Build the mandatory partner_id predicate for a resolved scope."""
    ids = ", ".join(str(int(i)) for i in allowed_partner_ids)
    return f"partner_id IN ({ids})"

def _format_conversation_history(history: list, max_turns: int = 2) -> str:
    """
    Format conversation history for LLM injection.
    Limits to `max_turns` (e.g., last 2 user-assistant exchanges = 4 messages).
    Strips out markdown tables from assistant responses to save tokens.
    """
    if not history:
        return "No previous history."
        
    recent_history = history[-(max_turns * 2):]
    formatted_msgs = []
    
    for msg in recent_history:
        role = msg.get("role", "user") if isinstance(msg, dict) else "user"
        content = msg.get("content", "") if isinstance(msg, dict) else str(msg)
        
        if role == "assistant":
            # Strip markdown tables (lines starting and ending with '|')
            content = "\n".join(
                line for line in content.split("\n") 
                if not (line.strip().startswith("|") and line.strip().endswith("|"))
            )
            content = re.sub(r'\n{3,}', '\n\n', content).strip()
            if not content:
                content = "(Data table omitted for brevity)"
                
        formatted_msgs.append(f"{role.capitalize()}: {content}")
        
    return "\n".join(formatted_msgs)



# ---------------------------------------------------------------------------
# Intent classifier + query planner — single LLM call, single round-trip
# ---------------------------------------------------------------------------
# Merging these two concerns into one agent saves a full LLM round-trip and
# one JSON parse.  The model classifies intent AND decides whether the query
# needs decomposition into independent sub-queries — all in ~200 tokens.
# ---------------------------------------------------------------------------

_INTENT_AND_PLANNER_PROMPT = """You are a query classifier, planner, and requirement completeness checker for a finance AI assistant.

Step 1 — Classify the user's message into exactly one of:
  - finance_query : asks about invoices, payments, revenue, expenses, journals, ledger, KPIs, budget, or any financial data. If the user is responding to a previous clarification question with a date/number/parameter (e.g. "2025", "Jul"), it is a continuation of a finance_query, so classify it as finance_query!
  - ambiguous     : partially finance-related but needs clarification
  - off_topic     : greetings, small talk, unrelated questions, or gibberish

Step 2 — For a finance_query, perform a Requirement Completeness and Feasibility Check.
Determine if you have enough information AND if the requested data falls within the Available Data Window.
Today's Date: {today_date}
Available Data Window: {data_window}
Authorized Scope: {scope_rules_text}

If critical information is missing (e.g., missing time period, missing metric), OR if the user asks for data outside the Available Data Window (e.g., future dates like "next month", or dates before the window starts), set clarification_needed to true and provide a targeted clarification_question. 
If the user asks for out-of-range dates, the clarification_question MUST explicitly inform them of the Available Data Window and ask if they want to query the available dates instead.
If the user explicitly asks for data about an organization/company that is NOT in their Authorized Scope, set clarification_needed to true and explicitly warn them that they are not authorized to view that organization's data. Do not generate sub_queries for unauthorized organizations.

CRITICAL RULE FOR TIME PERIODS: Unless the user explicitly says "all time" or "ever", a specific time period (e.g., month, quarter, year, or exact date range) is ABSOLUTELY REQUIRED for all financial queries. If the user does not specify a time period in the current query OR in the immediate conversation history, you MUST set clarification_needed to true and ask them what time period they want to analyze. Do NOT assume "all time" by default. Check the conversation history carefully; if they just clarified the organization, they STILL need to clarify the time period!

CRITICAL RULE FOR ORGANIZATIONS: If the user explicitly asks for a specific organization that IS within their Authorized Scope, you MUST NOT ask for confirmation. You MUST NOT set clarification_needed to true. Accept the request silently and generate the plan. IGNORE any past conversation history where you might have previously asked for confirmation—do NOT repeat that mistake!

CRITICAL RULE FOR CURRENCIES: ONLY ask for currency clarification if the user is asking about specific transaction-level data (e.g., individual invoices, bills, or payments for a specific customer/supplier) where transactional tables will be used, as these tables store multiple currencies (base vs foreign). DO NOT ask for clarification for high-level or aggregated metrics (e.g., "total sales invoice amount", "total revenue", "overall expenses") because these use the `kpi_base_metrics` table which only has a single base currency column. For KPI/aggregated queries, implicitly assume Base Currency and DO NOT set clarification_needed to true.

Step 3 — If the query is complete and feasible (clarification_needed=false), decide if the query needs decomposition.
A query needs decomposition (is_compound=true) ONLY when it requires data from 2+ truly
independent sources that cannot be answered by a single SQL query/JOIN.
DO NOT decompose queries that just ask for multiple time periods or multiple entities from the SAME domain (e.g. "Sales in 2025 and 2026" or "Revenue for Jan and Feb" MUST be one query, is_compound=false).

Decomposition examples:
  "Revenue vs expenses this month" → sq1: revenue from kpi_base_metrics, sq2: OPEX+COGS from kpi_base_metrics
  "Unpaid invoices and payments made by each customer" → sq1: unpaid invoices, sq2: customer payments
  "Total purchases and payments per supplier" → sq1: purchase_invoice totals, sq2: supplier_payment totals

Available data sources:
  - kpi_base_metrics    : pre-aggregated revenue, COGS, OPEX, AR, AP, cash, GST, TDS per period
  - sales_invoice       : individual customer invoice transactions
  - purchase_invoice    : individual supplier bill transactions
  - customer_payment    : customer receipt transactions
  - supplier_payment    : vendor payment transactions
  - journal             : GL journal entries
  - ledger_balance      : trial balance / ledger data
  - budget              : annual budget allocations by company, ledger, category, and segment

Return ONLY valid JSON (no markdown, no extra text):
{
  "intent": "<finance_query|ambiguous|off_topic>",
  "reason": "<one sentence>",
  "clarification_needed": true or false,
  "clarification_question": "<the follow up question to ask the user, if needed>",
  "is_compound": true or false,
  "sub_queries": [
    {"id": "sq1", "description": "what this fetches", "focus": "which table/domain"}
  ]
}

Constraints:
- If intent is NOT finance_query → clarification_needed must be false, is_compound false, sub_queries [].
- If clarification_needed is true → is_compound must be false, sub_queries [].
- If intent is finance_query but answerable with one query → is_compound false, sub_queries [].
- Maximum 3 sub-queries; consolidate if more would be needed.
- SECURITY RULE: The clarification_question MUST NEVER expose internal database table names (like 'sales_invoice'), internal column names, internal tenant IDs, or internal organization IDs. Speak purely in natural business terms. If an action is forbidden, simply say you don't have access, without revealing the underlying ID numbers."""

async def intent_and_planner_agent(orchestrator, state: AgentState) -> AgentState:
    """Classify intent and plan sub-queries in a single LLM call."""
    user_query = state["user_query"].strip()

    history = state.get("history", [])
    history_text = _format_conversation_history(history)
    
    # Early load of data window to prevent impractical out-of-range queries
    try:
        min_d, max_d = await orchestrator.get_data_window()
        try:
            from datetime import datetime
            min_obj = datetime.strptime(min_d, "%Y-%m-%d")
            max_obj = datetime.strptime(max_d, "%Y-%m-%d")
            data_window = f"{min_obj.strftime('%B %d, %Y')} to {max_obj.strftime('%B %d, %Y')}"
        except Exception:
            data_window = f"{min_d} to {max_d}"
    except Exception as e:
        logger.warning("[intent_and_planner] Failed to get data window: %s", e)
        data_window = "Unknown"

    allowed_partner_ids = state.get("allowed_partner_ids") or []
    scope_mode = state.get("scope_mode", "single")
    org_names = state.get("org_names", {})

    if scope_mode == "single":
        auth_orgs = ", ".join(f"{pid} ({org_names.get(str(pid), org_names.get(pid, 'Unknown'))})" for pid in allowed_partner_ids)
        scope_rules_text = f"SINGLE ORGANIZATION. The user is ONLY authorized to query data for: {auth_orgs}. They CANNOT access any other organization."
    else:
        auth_orgs = ", ".join(f"{pid} ({org_names.get(str(pid), org_names.get(pid, 'Unknown'))})" for pid in allowed_partner_ids)
        scope_rules_text = f"CONSOLIDATED TENANT. The user is authorized to query data across these organizations: {auth_orgs}."

    today_date = date.today().isoformat()
    prompt = _INTENT_AND_PLANNER_PROMPT.replace("{data_window}", str(data_window)).replace("{today_date}", today_date).replace("{scope_rules_text}", scope_rules_text)
    prompt += f"\n\nConversation History:\n{history_text}\n\nUser message: {user_query}"
    
    try:
        resp = await asyncio.wait_for(
            orchestrator.json_llm.ainvoke([HumanMessage(content=prompt)]),
            timeout=30.0
        )
        raw = resp.content.strip()
        logger.info(f"[DEBUG intent] raw response: {raw}")
        raw = re.sub(r"(^```json\s*\n?)|(\n?\s*```$)", "", raw, flags=re.MULTILINE)
        data = json.loads(raw)
        intent = data.get("intent", "finance_query")
        clarification_needed = data.get("clarification_needed", False)
        clarification_question = data.get("clarification_question", "")
        is_compound = data.get("is_compound", False) if intent == "finance_query" and not clarification_needed else False
        sub_queries = data.get("sub_queries", []) if is_compound else []
    except Exception as e:
        logger.warning("[intent_and_planner] failed, defaulting to finance_query/simple: %s", e)
        intent = "finance_query"
        clarification_needed = False
        clarification_question = ""
        is_compound = False
        sub_queries = []

    state["intent"] = intent
    state["clarification_needed"] = clarification_needed
    state["clarification_question"] = clarification_question
    state["is_compound"] = is_compound
    state["sub_query_plan"] = sub_queries if (is_compound and len(sub_queries) >= 2) else []
    state["sub_query_results"] = []

    if intent == "off_topic":
        state["error"] = "off_topic"
        state["natural_language_answer"] = (
            "I'm built specifically for finance queries — invoices, payments, revenue, "
            "expenses, ledger balances, and similar. Could you ask me something along those lines?"
        )
    elif intent == "ambiguous":
        state["error"] = "ambiguous"
        state["natural_language_answer"] = (
            "Could you be more specific? I can help with things like sales invoices, "
            "purchase bills, customer payments, supplier payments, or GL journal entries."
        )
    elif is_compound and len(sub_queries) >= 2:
        logger.info(
            "[intent_and_planner] compound query detected, %d sub-queries: %s",
            len(sub_queries), [sq["description"] for sq in sub_queries],
        )

    logger.info(
        "[intent_and_planner] intent=%s is_compound=%s query=%r",
        intent, state["is_compound"], user_query,
    )
    return state


async def schema_discovery_agent(orchestrator, state: AgentState) -> AgentState:
    """Load or discover database schema."""
    logger.debug("[schema_discovery] checking schema...")

    if orchestrator.cached_schema and orchestrator.cached_relationships:
        logger.debug("[schema_discovery] using pre-loaded schema metadata")
        state["schema_metadata"] = {
            "tables": orchestrator.cached_schema,
            "relationships": orchestrator.cached_relationships,
            "cached": True
        }
        state["messages"].append(AIMessage(content="Schema loaded from memory"))
        return state

    cached_schema = load_metadata(orchestrator.schema_file)
    cached_relationships = load_metadata(orchestrator.relationships_file)

    if cached_schema and cached_relationships:
        logger.debug("[schema_discovery] using cached schema metadata from disk")
        orchestrator.cached_schema = cached_schema
        orchestrator.cached_relationships = cached_relationships
        state["schema_metadata"] = {
            "tables": cached_schema,
            "relationships": cached_relationships,
            "cached": True
        }
        state["messages"].append(AIMessage(content="Schema loaded from disk cache"))
        return state

    try:
        logger.info("[schema_discovery] cache missing, executing discovery from DB...")
        from .schema import discover_schema_from_db
        schema, rels = await discover_schema_from_db(
            orchestrator.pool, orchestrator.schema_file, orchestrator.relationships_file
        )
        orchestrator.cached_schema = schema
        orchestrator.cached_relationships = rels

        state["schema_metadata"] = {
            "tables": schema,
            "relationships": rels,
            "cached": False
        }
        count = len(schema) if schema else 0
        state["messages"].append(AIMessage(content=f"Schema discovered: {count} tables"))

    except Exception as e:
        logger.exception("[schema_discovery] discovery failed: %s", e)
        state["error"] = f"Schema discovery failed: {str(e)}"
        state["messages"].append(AIMessage(content=f"Error in schema discovery: {str(e)}"))

    return state


async def table_description_agent(orchestrator, state: AgentState) -> AgentState:
    """Generate or load AI descriptions for each table."""
    logger.debug("[table_description] running...")

    if orchestrator.cached_descriptions and len(orchestrator.cached_descriptions) > 0:
        logger.debug("[table_description] using in-memory cached descriptions")
        state["schema_metadata"]["table_descriptions"] = orchestrator.cached_descriptions
        return state

    descriptions_from_disk = load_metadata(orchestrator.table_descriptions_file)
    if descriptions_from_disk and len(descriptions_from_disk) > 0:
        logger.debug("[table_description] loaded descriptions from disk")
        orchestrator.cached_descriptions = descriptions_from_disk
        state["schema_metadata"]["table_descriptions"] = descriptions_from_disk
        return state

    logger.info("[table_description] no descriptions found anywhere, generating with LLM (once-only)")
    schema = state["schema_metadata"].get("tables", {})
    if not schema:
        logger.warning("[table_description] no schema available, skipping description generation")
        state["schema_metadata"]["table_descriptions"] = {}
        return state

    descriptions = {}
    for table, details in schema.items():
        columns = details.get("columns", {})
        foreign_keys = details.get("foreign_keys", [])
        col_list = [f"{name} ({info['type']})" for name, info in columns.items()]
        fk_list = [f"{fk['column']} → {fk['references_table']}.{fk['references_column']}" for fk in foreign_keys]

        llm_prompt = f"""You are a database documentation assistant.
Create a SHORT description (1–2 sentences) for the table below.

Table Name: {table}
Columns: {chr(10).join(col_list)}
Foreign Keys: {chr(10).join(fk_list) if fk_list else 'None'}

Return ONLY valid JSON:
{{"table": "{table}", "description": "<short description>"}}"""

        try:
            resp = await asyncio.wait_for(
                orchestrator.json_llm.ainvoke([HumanMessage(content=llm_prompt)]),
                timeout=15.0
            )
            raw = resp.content.strip()
            raw = re.sub(r"(^```json\s*\n?)|(\n?\s*```$)", "", raw.strip(), flags=re.MULTILINE)
            data = json.loads(raw)
            desc = data.get("description", "No description available.").strip()
            descriptions[table] = desc
            logger.debug("[table_description] %s: %s", table, desc)
        except Exception as e:
            logger.warning("[table_description] failed for table %s: %s", table, e)
            descriptions[table] = "No description available."

    save_metadata(orchestrator.table_descriptions_file, descriptions)
    orchestrator.cached_descriptions = descriptions
    state["schema_metadata"]["table_descriptions"] = descriptions

    logger.info("[table_description] generated and cached %d descriptions", len(descriptions))
    return state


# ---------------------------------------------------------------------------
# KPI fast-path — answer period-summary queries straight from kpi_base_metrics
# ---------------------------------------------------------------------------

KPI_METRIC_KEYWORDS: dict[str, list[str]] = {
    "revenue":                    ["revenue", "sales", "income from operations", "top line", "turnover"],
    "cogs":                       ["cogs", "cost of goods", "cost of sales", "cost of revenue"],
    "opex":                       ["opex", "operating expense", "operating cost", "overhead"],
    "other_income":               ["other income", "non-operating income", "miscellaneous income"],
    "other_expense":              ["other expense", "miscellaneous expense"],
    "interest_expense":           ["interest expense", "finance cost", "finance charges", "borrowing cost"],
    "tax_expense":                ["tax", "income tax", "tax expense", "provision for tax"],
    "depreciation":               ["depreciation"],
    "amortization":               ["amortization", "amortisation"],
    "cash_balance":               ["cash", "cash balance", "bank balance", "closing cash"],
    "ar_outstanding":             ["accounts receivable", "ar", "receivables", "debtors", "trade receivables", "outstanding receivables"],
    "ap_outstanding":             ["accounts payable", "ap", "payables", "creditors", "trade payables", "outstanding payables"],
    "equity":                     ["equity", "net worth", "shareholders equity", "owners equity"],
    "gst_input":                  ["gst input", "input tax credit", "itc", "input gst"],
    "gst_output":                 ["gst output", "output tax", "output gst"],
    "tds_payable":                ["tds payable", "tds liability"],
    "tds_receivable":             ["tds receivable", "tds asset"],
    "cash_in":                    ["cash inflow", "cash in", "receipts"],
    "cash_out":                   ["cash outflow", "cash out", "disbursements"],
    "invoice_count":              ["invoice count", "number of invoices", "how many invoices", "sales invoice count"],
    "bill_count":                 ["bill count", "number of bills", "how many bills", "purchase bill count"],
    "customer_payment_count":     ["customer payment count", "number of customer payments", "receipt count"],
    "supplier_payment_count":     ["supplier payment count", "number of supplier payments", "vendor payment count"],
    "other_current_assets":       ["other current assets", "current assets", "short term assets"],
    "fixed_assets":               ["fixed assets", "non-current assets", "property plant equipment", "ppe", "long term assets"],
    "other_current_liabilities":  ["other current liabilities", "current liabilities", "short term liabilities"],
    "long_term_liabilities":      ["long term liabilities", "non-current liabilities", "long term debt"],
    "deferred_revenue":           ["deferred revenue", "unearned revenue", "advance from customers"],
    "profit":                     ["profit", "net profit", "net income", "earnings", "bottom line", "pbt", "pat"],
}


def _kpi_columns_for_query(user_query: str) -> list[str]:
    """Return matched kpi_base_metrics columns for the query, empty if no match."""
    q = user_query.lower()
    # Budget queries must go to finance.budget, not kpi_base_metrics
    if "budget" in q:
        return []
    matched = []
    for col, keywords in KPI_METRIC_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            matched.append(col)
    return matched


def _posting_period_cascading_formats(user_query: str) -> list[str]:
    """Generate multiple posting_period format variations for cascading search."""
    import re
    text = user_query.lower()
    formats = []
    
    # Try to extract month and year patterns
    month_map = {
        "january": "01", "february": "02", "march": "03", "april": "04",
        "may": "05", "june": "06", "july": "07", "august": "08",
        "september": "09", "october": "10", "november": "11", "december": "12",
        "jan": "01", "feb": "02", "mar": "03", "apr": "04",
        "jun": "06", "jul": "07", "aug": "08",
        "sep": "09", "oct": "10", "nov": "11", "dec": "12",
    }
    
    # Extract month name + year
    pattern = r"(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)[.\s-]*(\d{4}|\d{2})"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        month_name = match.group(1).lower()
        year_str = match.group(2)
        
        # Normalize year to 2-digit format
        year_2digit = year_str[-2:] if len(year_str) == 4 else year_str
        year_4digit = year_str if len(year_str) == 4 else f"20{year_str}"
        
        month_abbr = month_name[:3] if len(month_name) > 3 else month_name
        month_abbr_cap = month_abbr.capitalize()
        month_num = month_map.get(month_abbr, "")
        
        # Generate all possible format variations
        if month_num:
            # Format variations to try
            variations = [
                f"{month_abbr_cap} {year_4digit}",  # May 2026
                f"{month_abbr_cap}-{year_2digit}",  # May-26
                f"{month_num}-{month_abbr_cap}",     # 05-May
                f"{year_2digit}-{month_abbr_cap}",   # 26-May
                f"{month_abbr_cap.lower()}-{year_2digit}",  # may-26
                f"{year_4digit}-{month_num}",    # 2026-05
                f"{month_num}/{year_2digit}",    # 05/26
                f"{year_2digit}/{month_num}",    # 26/05
            ]
            formats.extend(variations)
    
    return formats or []


async def kpi_fastpath_agent(orchestrator, state: AgentState) -> AgentState:
    """
    Attempt to answer the query directly from kpi_base_metrics.
    Sets state['kpi_fastpath_hit'] = True and populates execution_result if successful.
    Falls through (kpi_fastpath_hit = False) if the KPI table cannot answer the query.
    """
    user_query = state["user_query"]
    allowed_partner_ids = state.get("allowed_partner_ids") or []
    partner_filter = _partner_filter_sql(allowed_partner_ids)
    min_date, max_date = await orchestrator.get_data_window()

    # Resolve relative date expressions to concrete posting_period labels.
    # This prevents the LLM from using unreliable ORDER BY refreshed_at for
    # "current month" / "last month" queries where no explicit date is given.
    _max_date_obj = date.fromisoformat(max_date)
    _anchor = date.today()
    _period_hint = resolve_posting_period_hint(
        user_query, anchor_date=_anchor, max_date=_max_date_obj
    )
    period_hint_section = (
        f"\n### Resolved Date Context (CRITICAL — use this, not refreshed_at)\n"
        f"  Today: {_anchor.isoformat()} | Latest data available: {max_date}\n"
        f"  Resolved period for this query: {_period_hint}\n"
        f"  Use posting_period ILIKE with the resolved period above.\n"
        f"  NEVER use ORDER BY refreshed_at to determine current/latest period.\n"
    ) if _period_hint else ""

    history = state.get("history", [])
    history_text = _format_conversation_history(history)

    matched_cols = _kpi_columns_for_query(user_query)
    if not matched_cols:
        logger.debug("[kpi_fastpath] no KPI columns matched, skipping")
        state["kpi_fastpath_hit"] = False
        return state

    col_list = ", ".join(matched_cols)
    
    # Add profit calculation guidance if profit was requested
    profit_guidance = ""
    if "profit" in matched_cols:
        profit_guidance = """
For profit calculations, use this formula:
  profit = revenue + other_income - cogs - opex - other_expense - interest_expense - tax_expense - depreciation - amortization
  
Wrap each component in COALESCE(..., 0) to handle NULL values, then compute the final profit.
"""
    
    prompt = f"""You are a PostgreSQL expert. Build a query against finance.kpi_base_metrics only.

Available columns:
  id, tenant_id, partner_id, company, posting_period, subsidiary_currency,
  revenue, cogs, opex, other_income, other_expense, interest_expense, tax_expense,
  depreciation, amortization, cash_balance, ar_outstanding, ap_outstanding,
  other_current_assets, fixed_assets, other_current_liabilities, long_term_liabilities,
  equity, deferred_revenue, gst_input, gst_output, tds_payable, tds_receivable,
  cash_in, cash_out, invoice_count, bill_count,
  customer_payment_count, supplier_payment_count, refreshed_at.

### Finance Term → Column Mapping (CRITICAL)
Use these mappings when the user uses finance terminology:
  - "revenue" / "sales" / "turnover" / "top line" / "income from operations" → revenue
  - "COGS" / "cost of goods" / "cost of sales" / "cost of revenue"           → cogs
  - "OPEX" / "operating expense" / "operating cost" / "overhead"             → opex
  - "gross profit"                                                            → revenue - cogs
  - "EBIT" / "operating profit"                                               → revenue - cogs - opex
  - "net profit" / "profit" / "net income" / "PAT" / "bottom line"           → revenue + other_income - cogs - opex - other_expense - interest_expense - tax_expense - depreciation - amortization
  - "accounts receivable" / "AR" / "debtors" / "receivables"                 → ar_outstanding
  - "accounts payable" / "AP" / "creditors" / "payables"                     → ap_outstanding
  - "cash" / "cash balance" / "bank balance"                                  → cash_balance
  - "cash inflow" / "receipts"                                               → cash_in
  - "cash outflow" / "disbursements"                                         → cash_out
  - "equity" / "net worth" / "shareholders equity"                            → equity
  - "fixed assets" / "PPE" / "non-current assets"                            → fixed_assets
  - "current assets"                                                          → cash_balance + ar_outstanding + other_current_assets
  - "current liabilities"                                                     → ap_outstanding + other_current_liabilities
  - "deferred revenue" / "unearned revenue" / "advance from customers"        → deferred_revenue
  - "input GST" / "ITC" / "input tax credit"                                 → gst_input
  - "output GST" / "output tax"                                               → gst_output
  - "TDS payable"                                                             → tds_payable
  - "TDS receivable"                                                          → tds_receivable
  - "interest expense" / "finance cost" / "finance charges"                  → interest_expense
  - "tax expense" / "income tax" / "provision for tax"                       → tax_expense

### Date Filtering (CRITICAL)
  - `posting_period` is a TEXT label. The database ALWAYS stores the month as a 3-letter abbreviation (e.g. 'Jan', 'Feb', 'Mar', 'Jul'). You MUST match using the 3-letter month abbreviation!
  - DO NOT use `transaction_date` — this column does NOT exist in kpi_base_metrics.
  - DO NOT use date math functions like DATE_TRUNC or EXTRACT on posting_period.
  - For a specific month: posting_period ILIKE '%May%2026%' or '%Jul%2026%' (DO NOT use full month names like '%July%').
  - For a range of months: use OR with ILIKE for each month
  - For the latest period when no date hint is given: ORDER BY posting_period DESC LIMIT 1 per company
  - NEVER use ORDER BY refreshed_at — it is a row-level update timestamp, not the period date.
{period_hint_section}
### Few-Shot Examples
-- Q: "What is the revenue for May 2026?"
SELECT company, posting_period, revenue
FROM finance.kpi_base_metrics
WHERE {partner_filter}
  AND posting_period ILIKE '%May%2026%';

-- Q: "Show me COGS and OPEX for the current period" (resolved: Jul 2026)
SELECT company, posting_period, cogs, opex
FROM finance.kpi_base_metrics
WHERE {partner_filter}
  AND posting_period ILIKE '%Jul%2026%';

-- Q: "What is accounts receivable outstanding?" (resolved: Jul 2026)
SELECT company, posting_period, ar_outstanding
FROM finance.kpi_base_metrics
WHERE {partner_filter}
  AND posting_period ILIKE '%Jul%2026%';

-- Q: "Calculate net profit for Q1 2026"
SELECT company, posting_period,
  (COALESCE(revenue,0) + COALESCE(other_income,0)
   - COALESCE(cogs,0) - COALESCE(opex,0) - COALESCE(other_expense,0)
   - COALESCE(interest_expense,0) - COALESCE(tax_expense,0)
   - COALESCE(depreciation,0) - COALESCE(amortization,0)) AS net_profit
FROM finance.kpi_base_metrics
WHERE {partner_filter}
  AND (posting_period ILIKE '%Jan%2026%' OR posting_period ILIKE '%Feb%2026%' OR posting_period ILIKE '%Mar%2026%');

-- Q: "Show cash balance and equity trend"
SELECT company, posting_period, cash_balance, equity
FROM finance.kpi_base_metrics
WHERE {partner_filter}
ORDER BY posting_period DESC;

{profit_guidance}

Tenant filter (MANDATORY on every query): {partner_filter}
Data window: {min_date} to {max_date}.

Conversation History:
{history_text}

Latest User query: {user_query}

Return ONLY the raw SQL, no markdown, no explanation.
If the query cannot be answered from kpi_base_metrics alone, return exactly: CANNOT_ANSWER"""

    try:
        resp = await asyncio.wait_for(
            orchestrator.sql_llm.ainvoke([HumanMessage(content=prompt)]),
            timeout=30.0
        )
        raw_sql = resp.content.strip()
        if "```sql" in raw_sql:
            raw_sql = raw_sql.split("```sql")[1].split("```")[0].strip()
        elif "```" in raw_sql:
            raw_sql = raw_sql.split("```")[1].split("```")[0].strip()
        sql = raw_sql

        if sql == "CANNOT_ANSWER" or not sql.upper().lstrip().startswith(("SELECT", "WITH")):
            logger.info("[kpi_fastpath] LLM said CANNOT_ANSWER or invalid SQL, falling through")
            state["kpi_fastpath_hit"] = False
            return state

        is_safe, safety_err = validate_sql_safety(sql)
        if not is_safe:
            logger.warning("[kpi_fastpath] safety check failed: %s", safety_err)
            state["kpi_fastpath_hit"] = False
            return state

        scope_ok, _ = enforce_partner_scope(sql, allowed_partner_ids, {"kpi_base_metrics"})
        if not scope_ok:
            logger.warning("[kpi_fastpath] scope check failed, falling through")
            state["kpi_fastpath_hit"] = False
            return state

        async with orchestrator.pool.acquire() as conn:
            rows = await asyncio.wait_for(conn.fetch(sql), timeout=15.0)

            # If the query returned 0 rows, try cascading posting_period formats
            if not rows:
                logger.debug("[kpi_fastpath] initial query returned 0 rows, trying cascading formats...")
                cascading_formats = _posting_period_cascading_formats(user_query)
                
                for fmt in cascading_formats:
                    # Build a cascading query that uses ILIKE to match the posting_period
                    cascading_sql = re.sub(
                        r"posting_period\s+IN\s*\([^)]+\)",
                        f"posting_period ILIKE '%{fmt}%'",
                        sql,
                        flags=re.IGNORECASE
                    )
                    
                    try:
                        cascade_rows = await asyncio.wait_for(conn.fetch(cascading_sql), timeout=15.0)
                        if cascade_rows:
                            rows = cascade_rows
                            sql = cascading_sql
                            logger.info("[kpi_fastpath] cascading format matched: %s", fmt)
                            break
                    except Exception as e:
                        logger.debug("[kpi_fastpath] cascading format %s failed: %s", fmt, e)
                        continue

        if rows:
            result_rows = [dict(r) for r in rows]

            # all_null guard: SUM() returns a single NULL row for periods with no data in
            # kpi_base_metrics.  Treat that as a miss so we fall through to the transactional
            # tables — returning $0.00 for a period that was never computed would be wrong.
            all_null = all(
                v is None
                for row in result_rows
                for v in row.values()
            )
            if all_null:
                state["kpi_fastpath_hit"] = False
                logger.info("[kpi_fastpath] all values NULL (period not in kpi_base_metrics), falling through")
                return state

            state["execution_result"] = {
                "success": True,
                "rows": result_rows,
                "row_count": len(result_rows),
                "columns": list(result_rows[0].keys()),
                "source": "kpi_base_metrics",
            }
            state["generated_sql"] = sql
            state["kpi_fastpath_hit"] = True
            logger.info("[kpi_fastpath] HIT — answered from kpi_base_metrics, rows=%d", len(result_rows))
        else:
            # kpi_base_metrics returned 0 rows — fall through to full transactional search
            state["kpi_fastpath_hit"] = False
            logger.info("[kpi_fastpath] 0 rows from kpi_base_metrics, falling through to full path")

    except Exception as e:
        logger.warning("[kpi_fastpath] failed, falling through: %s", e)
        state["kpi_fastpath_hit"] = False

    return state


async def table_selection_agent(orchestrator, state: AgentState) -> AgentState:
    """LLM-based table selection for the user query."""
    logger.debug("[table_selection] analyzing query for relevant tables (LLM-based)...")
    user_query = state["user_query"]
    all_tables = list(state["schema_metadata"]["tables"].keys())
    table_descriptions = state["schema_metadata"].get("table_descriptions", {})

    tables_with_desc = []
    for t in all_tables:
        desc = table_descriptions.get(t, "").strip()
        if desc:
            tables_with_desc.append(f"{t}: {desc}")
        else:
            tables_with_desc.append(t)

    tables_block = "\n".join(tables_with_desc)
    logger.debug("[table_selection] passing %d tables with descriptions to LLM...", len(all_tables))

    history = state.get("history", [])
    history_text = _format_conversation_history(history)

    prompt = f"""
    Given the following database tables with their descriptions, identify which tables are most relevant to answer the user query.
        Database Context: {orchestrator.db_config.get('description', '')}

        Conversation History:
        {history_text}
        
        Latest User Query: {user_query}

        Available Tables:
        {tables_block}

        Consider:
        1. Direct mentions of entities in the query
        2. Related entities that might be needed for joins
        3. Tables that contain the requested information

        Return a JSON object with this structure:
        {{"selected_tables": ["table1", "table2"], "reasoning": "brief explanation"}}
    """

    try:
        response = await asyncio.wait_for(
            orchestrator.json_llm.ainvoke([HumanMessage(content=prompt)]),
            timeout=60.0
        )
        content = response.content.strip()
        if content.startswith("```"):
            parts = content.split("```")
            if len(parts) >= 2:
                content = parts[1].strip()
        if content.lower().startswith("json"):
            content = content[4:].strip()
        result = json.loads(content)
        selected_tables = result.get("selected_tables", [])
        selected_tables = [t for t in selected_tables if t in all_tables]

        if not selected_tables:
            logger.warning("[table_selection] no tables selected by LLM, falling back to keyword match")
            query_lower = user_query.lower()
            selected_tables = [t for t in all_tables if any(word in query_lower for word in t.split('.'))]
        if not selected_tables:
            selected_tables = all_tables[:3]

        logger.info("[table_selection] selected tables: %s", ', '.join(selected_tables))
        state["selected_tables"] = selected_tables
        state["messages"].append(AIMessage(content=f"Selected tables: {', '.join(selected_tables)}"))

    except Exception as e:
        logger.exception("[table_selection] error: %s", e)
        state["selected_tables"] = all_tables[:3]
        state["messages"].append(AIMessage(content=f"Error in table selection: {str(e)}"))

    return state


async def structure_capture_agent(orchestrator, state: AgentState) -> AgentState:
    """Capture detailed structure for selected tables."""
    logger.debug("[structure_capture] capturing table structures...")
    selected_tables = state["selected_tables"]
    schema_metadata = state["schema_metadata"]["tables"]
    table_structures = {}
    
    # Auditing/internal columns that don't need to be sent to the LLM (saves tokens)
    PRUNE_COLS = {"created_at", "updated_at", "created_by", "updated_by", "deleted", 
                  "sync_status", "idx", "metadata", "creation", "modified"}
                  
    for table in selected_tables:
        if table in schema_metadata:
            # Create a shallow copy and prune irrelevant columns
            struct = dict(schema_metadata[table])
            if "columns" in struct:
                pruned_cols = {
                    k: v for k, v in struct["columns"].items() 
                    if k.lower() not in PRUNE_COLS
                }
                struct["columns"] = pruned_cols
            table_structures[table] = struct
            logger.debug("[structure_capture] captured: %s", table)
            
    state["table_structures"] = table_structures
    state["messages"].append(AIMessage(content=f"Captured and pruned structures for {len(table_structures)} tables"))
    return state


async def sql_generation_agent(orchestrator, state: AgentState) -> AgentState:
    """Generate SQL query from natural language using LLM."""
    logger.debug("[sql_generation] generating SQL query...")
    user_query = state["user_query"]
    history = state.get("history", [])
    history_text = _format_conversation_history(history)
    table_structures = state["table_structures"]
    relationships = state["schema_metadata"]["relationships"]
    allowed_partner_ids = state.get("allowed_partner_ids") or []
    scope_mode = state.get("scope_mode", "single")
    org_names = state.get("org_names", {})

    if not allowed_partner_ids:
        return {
            "error": "Scope is missing (no allowed partner_ids). Cannot generate safe SQL.",
            "messages": [AIMessage(content="Error: Request scope not found.")]
        }

    partner_filter = _partner_filter_sql(allowed_partner_ids)
    min_date, max_date = await orchestrator.get_data_window()
    try:
        min_date_obj = datetime.strptime(min_date, "%Y-%m-%d").date()
        max_date_obj = datetime.strptime(max_date, "%Y-%m-%d").date()
        anchor = max_date_obj
        anchor_month = anchor.strftime("%B %Y")
        anchor_year = str(anchor.year)
    except Exception:
        min_date_obj = None
        max_date_obj = None
        anchor = None
        anchor_month = "the latest available month"
        anchor_year = "the latest available year"

    if scope_mode == "consolidated":
        org_lines = "\n".join(f"    * {pid} = {org_names.get(str(pid), org_names.get(pid, 'Unknown'))}" for pid in allowed_partner_ids)
        scope_rules = f"""### Scope: CONSOLIDATED (tenant-wide)
- This request spans MULTIPLE organizations that belong to one tenant.
- EVERY tenant-scoped table MUST be filtered with: {partner_filter}
- If the user asks for a specific organization within the tenant, ADD an extra condition (e.g. AND partner_id = X), but you MUST still include the mandatory {partner_filter}.
- The organizations included are:
{org_lines}
- Aggregate across all of them (SUM/COUNT/AVG as appropriate) to give an overall view.
- When a per-organization breakdown is useful, also GROUP BY partner_id so each organization can be identified.
- NEVER use any partner_id outside the list above."""
    else:
        scope_rules = f"""### Scope: SINGLE ORGANIZATION
- This request is limited to exactly ONE organization.
- EVERY tenant-scoped table MUST be filtered with: {partner_filter}
- NEVER return data from any other partner_id."""

    has_partner_id = any(
        'partner_id' in structure.get('columns', {})
        for structure in table_structures.values()
    )
    if not has_partner_id:
        logger.warning("[sql_generation] partner_id column not found in selected tables")

    schema_context = []
    for table, structure in table_structures.items():
        columns = ", ".join([f"{col} ({info['type']})" for col, info in structure['columns'].items()])
        schema_context.append(f"Table: {table}\nColumns: {columns}")
        if structure['foreign_keys']:
            fks = ", ".join([f"{fk['column']} -> {fk['references_table']}.{fk['references_column']}" for fk in structure['foreign_keys']])
            schema_context.append(f"Foreign Keys: {fks}")

    relevant_rels = [rel for rel in relationships if rel['from_table'] in table_structures and rel['to_table'] in table_structures]
    tables_text = "\n\n".join(schema_context)

    date_filter_hint = ""
    if anchor is not None and min_date_obj is not None and max_date_obj is not None:
        derived_filter = derive_relative_date_filter(
            user_query,
            anchor_date=anchor,
            min_date=min_date_obj,
            max_date=max_date_obj,
        )
        if derived_filter:
            date_filter_hint = f"\n### Deterministic relative-date filter\nUse this exact predicate in the WHERE clause when the request is relative: {derived_filter}\n"

    history = state.get("history", [])
    history_text = _format_conversation_history(history)

    prompt = f"""
You are an expert PostgreSQL query generator for a multi-tenant FINANCE application.

Conversation History:
{history_text}

Latest User request: {user_query}

### Mandatory Tenant Rules
- This is a multi-tenant system. EVERY query MUST filter by {partner_filter}.
- Apply the filter on the table that actually has the partner_id column.
- For JOINs, apply the partner_id filter on the primary tenant-scoped table (e.g. sales_invoice, purchase_invoice, journal, customer_payment, supplier_payment).
- Never return data from any partner_id outside the allowed set.

{scope_rules}

### Available Data (IMPORTANT)
- The finance tables contain data from {min_date} to {max_date} (inclusive).
{date_filter_hint}
- The LATEST available date is {max_date}, so the LATEST available month is {anchor_month}. Treat that as "current month" / "this month".
- Resolve ALL relative periods ("today", "this month", "last month", "this quarter", "last quarter", "this year", "year to date", "last N months/days", etc.) RELATIVE TO the latest available date {max_date} - NOT relative to the real calendar date.
- NEVER rely on CURRENT_DATE / NOW() for any date math - always derive the range from the data window above.
- If a requested period falls partly or fully outside {min_date} – {max_date}, clamp it to that window.
- General rules for deriving ranges from the anchor date {max_date}:
    * "this month"  = first day to last day of the anchor month.
    * "last month"  = the calendar month immediately before the anchor month.
    * "this quarter"/"last quarter" = the calendar quarter of / immediately before the anchor date.
    * "this year"/"year to date" = Jan 1 of the anchor year up to (and including) the anchor date.
    * "last N months/days" = counted backwards from the anchor date.

### Date Column Rules (CRITICAL)
- For transactional tables (sales_invoice, purchase_invoice, customer_payment, supplier_payment, journal, ledger_balance):
  ALWAYS use the `transaction_date` column (type DATE) with a half-open range.
  - Month: WHERE transaction_date >= 'YYYY-MM-01' AND transaction_date < 'YYYY-MM+1-01'
  - Year:  WHERE transaction_date >= 'YYYY-01-01' AND transaction_date <  'YYYY+1-01-01'
  DO NOT filter on `posting_period` in these tables.
- For `kpi_base_metrics` ONLY:
  `transaction_date` does NOT exist. Use `posting_period` (TEXT label).
  The database ALWAYS stores the month as a 3-letter abbreviation (e.g. 'Jan', 'Feb', 'Mar', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec').
  You MUST match using the 3-letter month abbreviation! (e.g. posting_period ILIKE '%Jul%2026%'). Do NOT use the full month name like '%July%'.
- For `budget` ONLY:
  `transaction_date` does NOT exist. Use the `year` column (TEXT) for period matching.
  The `year` column stores fiscal year labels (e.g. '2026', 'FY2026'). Match with ILIKE.
  When NO specific year or period is mentioned (e.g. "total budget", "show budget"),
  DEFAULT to the latest available year: year ILIKE '%{anchor_year}%'.
  For relative month-range queries (e.g. "last 3 months"), extract the distinct calendar years
  covered by that range and filter with: year ILIKE '%{anchor_year}%' OR year ILIKE '%{str(int(anchor_year) - 1) if anchor_year.isdigit() else anchor_year}%' as needed.
  NEVER use transaction_date or date math on this table.
  *** CRITICAL — DEDUPLICATION REQUIRED ***
  The budget table has ~350x duplicate rows per internal_id caused by repeated sync imports.
  A raw SUM(budget_amount) will be 350x too large. You MUST always use this CTE pattern:
    WITH dedup_budget AS (
        SELECT DISTINCT ON (internal_id)
            <required columns>
        FROM finance.budget
        WHERE {partner_filter}
          AND year ILIKE '%<year>%'
        ORDER BY internal_id, id
    )
    SELECT ... FROM dedup_budget ...

### Domain Synonyms
- "customer orders" / "orders" / "invoices billed" / "sales"   → finance.sales_invoice
- "vendor bills" / "purchases" / "supplier invoices"            → finance.purchase_invoice
- "customer receipts" / "customer payments" / "collections"     → finance.customer_payment
    * Hard-code explicit date literals (e.g. 'YYYY-MM-DD') based on the anchor date above.
- NEVER filter on `posting_period` for date ranges — `posting_period` is a text label (e.g. '25-Apr', 'Jan-26'). Always filter on `transaction_date` (DATE type) with half-open ranges:
    `transaction_date >= 'YYYY-MM-01' AND transaction_date < 'YYYY-(MM+1)-01'`

### Available Tables & Schema:
{tables_text}

### Relationships:
{json.dumps(relationships, indent=2)}

IMPORTANT: The above relationships include both foreign key relationships and manual logical joins.
- Use the provided relationship metadata to choose the correct join keys.
- Do not infer joins purely from matching column names when a documented relationship exists.
- For example, if there is a logical relationship from finance.sales_invoice.document_number to finance.customer_payment.applying_transaction, use that join condition.

    ### Conversation History:
    {history_text}
    
    ### User Query:
    {user_query}

### Rules:
- Return ONLY valid executable SQL — no markdown formatting (no ```sql), no explanation text.
- Use proper explicit JOIN syntax (INNER JOIN, LEFT JOIN, etc.) when combining tables.
- For any text search or partial match, use ILIKE with wildcards (%text%).
- Use meaningful table aliases when queries involve multiple tables.
- Always use fully qualified table names (schema.table).
- Always include the {partner_filter} filter on every tenant-scoped table.
"""

    previous_error = state.get("error", "")
    previous_sql = state.get("generated_sql", "")
    is_retry = bool(previous_error and previous_sql)
    if is_retry:
        prompt += f"""
    ### PREVIOUS ATTEMPT FAILED
    The following SQL query was generated but failed during execution:
    ```sql
    {previous_sql}
    ```
    Error Message: {previous_error}

    Please analyze the error and generate a CORRECTED query that fixes this issue.
    """
    else:
        prompt += "\n    Generate the PostgreSQL query now."

    new_retry_count = state.get("retry_count", 0) + (1 if is_retry else 0)

    try:
        response = await asyncio.wait_for(
            orchestrator.sql_llm.ainvoke([HumanMessage(content=prompt)]),
            timeout=60.0
        )
        content = response.content.strip()
        sql_query = content

        if "```sql" in sql_query:
            sql_query = sql_query.split("```sql")[1].split("```")[0].strip()
        elif "```" in sql_query:
            sql_query = sql_query.split("```")[1].split("```")[0].strip()

        sql_query = sql_query.rstrip(';').strip()
        if anchor is not None and min_date_obj is not None and max_date_obj is not None:
            sql_query = apply_relative_date_filter(
                sql_query,
                user_query,
                anchor_date=anchor,
                min_date=min_date_obj,
                max_date=max_date_obj,
            )

        if not any(sql_query.upper().startswith(kw) for kw in ['SELECT', 'WITH']):
            raise ValueError("Not a valid SQL query")

        logger.info(
            "SQL generated: retry_count=%d scope_mode=%s tenant_id=%r partner_ids=%s sql=%s",
            new_retry_count,
            state.get("scope_mode"),
            state.get("tenant_id"),
            state.get("allowed_partner_ids"),
            sql_query,
        )

        state["generated_sql"] = sql_query
        state["error"] = ""
        state["execution_result"] = {}
        state["retry_count"] = new_retry_count
        state["messages"].append(AIMessage(content="SQL query generated"))
        return state

    except Exception as e:
        logger.exception(
            "SQL generation failed: retry_count=%d scope_mode=%s tenant_id=%r",
            new_retry_count,
            state.get("scope_mode"),
            state.get("tenant_id"),
        )
        state["error"] = f"SQL generation failed: {str(e)}"
        state["retry_count"] = new_retry_count
        state["messages"].append(AIMessage(content=f"Error generating SQL: {str(e)}"))
        return state


async def review_agent(orchestrator, state: AgentState) -> AgentState:
    """Review and validate the generated SQL query."""
    logger.debug("[review] validating SQL query...")
    sql_query = state["generated_sql"]
    table_structures = state["table_structures"]
    allowed_partner_ids = state.get("allowed_partner_ids") or []
    partner_filter = _partner_filter_sql(allowed_partner_ids)

    columns_context = {}
    for table, structure in table_structures.items():
        columns_context[table] = list(structure['columns'].keys())

    min_date, max_date = await orchestrator.get_data_window()
    try:
        min_date_obj = datetime.strptime(min_date, "%Y-%m-%d").date()
        max_date_obj = datetime.strptime(max_date, "%Y-%m-%d").date()
        anchor = max_date_obj
    except Exception:
        min_date_obj = None
        max_date_obj = None
        anchor = None

    prompt = f"""Review this PostgreSQL query for correctness for a multi-tenant finance app.

SQL Query:
{sql_query}

Required scope for this request: every tenant-scoped table MUST be filtered with `{partner_filter}`.
Available Tables and Columns:
{json.dumps(columns_context, indent=2)}

Check for:
1. Non-existent column names or wrong table references.
2. Syntax errors.
3. Missing JOIN conditions.
4. Missing or wrong partner_id filter. The base filter MUST be exactly `{partner_filter}` on every tenant-scoped table (never unfiltered). If the user asked for a specific organization, it is OK to have an additional `AND partner_id = X` filter, but the base `{partner_filter}` MUST still be present.
- When querying amounts from transactional tables (like sales_invoice or purchase_invoice), use `gross_amount` or `amt` for the base/local currency, and use `amount_fx` for the foreign/transaction currency.
- NEVER attempt to manually multiply amounts by `exchange_rate` to do currency conversions in SQL. Always rely on the pre-calculated `amount_fx` or `gross_amount` columns.
5. Use of `posting_period` for date filtering on transactional tables (sales_invoice, purchase_invoice, customer_payment, supplier_payment, journal) - WRONG. Date filters must use the `transaction_date` DATE column with a half-open range:
       transaction_date >= 'YYYY-MM-01' AND transaction_date < 'YYYY-(MM+1)-01'.
   EXCEPTION: `kpi_base_metrics` does NOT have `transaction_date`. For that table ONLY, use `posting_period` with ILIKE (e.g. posting_period ILIKE '%May%2026%'). Do NOT flag posting_period usage on kpi_base_metrics as an error.
6. Use of CURRENT_DATE / NOW() for month math - also WRONG. Hard-code the date range using the available data window ({min_date} to {max_date}, latest available date = {max_date}). All relative periods must be derived from that anchor date, not the real calendar date.
7. String comparisons against `posting_period` like '2025-11' or 'November 2025' on transactional tables - must be rewritten to use `transaction_date`. (Again, kpi_base_metrics is exempt — posting_period ILIKE is correct there.)
8. For LEFT JOINs: do not move partner_id filters from the ON clause into the WHERE clause. Preserve LEFT JOIN semantics.

IMPORTANT - do NOT do any of the following:
- Do NOT add a `tenant_id` filter. Tenant isolation is enforced exclusively through `{partner_filter}`; adding `tenant_id` is unnecessary and will break execution.
- Do NOT introduce SQL parameter placeholders (`$1`, `$2`, `<current_tenant_id>`, `:<name>`, etc.). Every value in the query must be a hard-coded literal.

If anything in checks 1-8 is wrong, rewrite the query and return it in corrected_sql.

Return ONLY a JSON object:
{{"valid": true or false, "issues": ["list of issues found"], "corrected_sql": "corrected SQL if issues found, else null"}}
"""

    try:
        response = await asyncio.wait_for(
            orchestrator.json_llm.ainvoke([HumanMessage(content=prompt)]),
            timeout=60.0
        )
        content = response.content.strip()
        if content.startswith("```"):
            parts = content.split("```")
            if len(parts) >= 2:
                content = parts[1].strip()
        if content.lower().startswith("json"):
            content = content[4:].strip()
        validation_result = json.loads(content)

        if anchor is not None and min_date_obj is not None and max_date_obj is not None:
            sql_query = apply_relative_date_filter(
                sql_query,
                state["user_query"],
                anchor_date=anchor,
                min_date=min_date_obj,
                max_date=max_date_obj,
            )
            state["generated_sql"] = sql_query

        if not validation_result.get("valid", False):
            logger.warning(
                "SQL review found issues: issues=%s corrected=%s",
                validation_result.get("issues", []),
                bool(validation_result.get("corrected_sql")),
            )
            if validation_result.get("corrected_sql"):
                corrected_sql = validation_result["corrected_sql"]
                if anchor is not None and min_date_obj is not None and max_date_obj is not None:
                    corrected_sql = apply_relative_date_filter(
                        corrected_sql,
                        state["user_query"],
                        anchor_date=anchor,
                        min_date=min_date_obj,
                        max_date=max_date_obj,
                    )
                logger.info("SQL corrected by review agent: corrected_sql=%s", corrected_sql)
                state["generated_sql"] = corrected_sql
        else:
            logger.info("SQL review passed: no issues found")

        state["validation_result"] = validation_result
        state["messages"].append(AIMessage(content=f"Validation: {'Valid' if validation_result.get('valid') else 'Issues found and corrected'}"))

    except Exception as e:
        logger.warning("[review] validation error, proceeding with original query: %s", e)
        state["validation_result"] = {"valid": True, "issues": [], "warning": str(e)}

    return state


async def execution_agent(orchestrator, state: AgentState) -> AgentState:
    """Execute the SQL query against the database."""
    logger.debug("[execution] executing SQL query...")
    sql_query = state["generated_sql"]
    allowed_partner_ids = state.get("allowed_partner_ids") or []

    try:
        is_safe, safety_error = validate_sql_safety(sql_query)
        if not is_safe:
            logger.error(
                "SQL blocked by safety check: reason=%s sql=%s",
                safety_error, sql_query,
            )
            state["execution_result"] = {"success": False, "error": safety_error, "row_count": 0}
            state["error"] = safety_error
            state["messages"].append(AIMessage(content=f"SQL blocked: {safety_error}"))
            return state

        # Deterministic, parser-based tenant-scope enforcement (the real boundary).
        scope_ok, scope_error = enforce_partner_scope(
            sql_query,
            allowed_partner_ids,
            getattr(orchestrator, "tenant_scoped_tables", set()),
        )
        if not scope_ok:
            logger.error(
                "SQL blocked by scope enforcement: reason=%s partner_ids=%s sql=%s",
                scope_error, allowed_partner_ids, sql_query,
            )
            state["execution_result"] = {"success": False, "error": scope_error, "row_count": 0}
            state["error"] = f"Scope enforcement failed: {scope_error}"
            state["messages"].append(AIMessage(content=f"Scope violation: {scope_error}"))
            return state

        async with orchestrator.pool.acquire() as conn:
            async with conn.transaction():
                try:
                    rows = await asyncio.wait_for(
                        conn.fetch(sql_query),
                        timeout=30.0
                    )

                    if rows:
                        columns = list(rows[0].keys())
                        result_rows = [dict(row) for row in rows]
                    else:
                        columns = []
                        result_rows = []

                    result = {
                        "success": True,
                        "rows": result_rows,
                        "row_count": len(result_rows),
                        "columns": columns
                    }
                    logger.info(
                        "SQL executed: scope_mode=%s tenant_id=%r partner_ids=%s "
                        "row_count=%d sql=%s",
                        state.get("scope_mode"),
                        state.get("tenant_id"),
                        allowed_partner_ids,
                        len(result_rows),
                        sql_query,
                    )
                    state["execution_result"] = result
                    state["messages"].append(AIMessage(content=f"Query executed: {len(result_rows)} rows returned"))

                    # Save to history
                    await orchestrator.save_query_history({
                        "scope_mode": state.get("scope_mode"),
                        "tenant_id": state.get("tenant_id"),
                        "allowed_partner_ids": state.get("allowed_partner_ids"),
                        "user_query": state["user_query"],
                        "sql_query": sql_query,
                        "row_count": len(result_rows),
                        "success": True,
                    })

                except asyncio.TimeoutError:
                    raise TimeoutError("Query execution timed out after 30 seconds. Try a more specific query.")

    except Exception as e:
        logger.exception(
            "SQL execution failed: scope_mode=%s tenant_id=%r partner_ids=%s sql=%s",
            state.get("scope_mode"),
            state.get("tenant_id"),
            allowed_partner_ids,
            sql_query,
        )
        state["execution_result"] = {"success": False, "error": str(e), "row_count": 0}
        state["error"] = f"Execution failed: {str(e)}"
        state["messages"].append(AIMessage(content=f"Execution error: {str(e)}"))

    return state


async def relationship_manager_agent(orchestrator, state: AgentState) -> AgentState:
    """Find and select related tables to expand query scope."""
    logger.debug("[relationship_manager] analyzing relationships...")
    selected_tables = state["selected_tables"]
    relationships = state["schema_metadata"]["relationships"]
    table_descriptions = state["schema_metadata"].get("table_descriptions", {})

    related_tables = set(selected_tables)
    for rel in relationships:
        if rel['from_table'] in selected_tables:
            related_tables.add(rel['to_table'])
        if rel['to_table'] in selected_tables:
            related_tables.add(rel['from_table'])

    candidate_tables = list(related_tables - set(selected_tables))
    if not candidate_tables:
        logger.debug("[relationship_manager] no additional related tables found")
        state["related_tables"] = []
        return state

    candidate_tables = candidate_tables[:10]
    user_query = state["user_query"]

    candidates_block = "\n".join(f"- {t}: {table_descriptions.get(t, 'No description available.')}" for t in candidate_tables)
    selected_block = ", ".join(selected_tables)

    prompt = f"""
        The initial SQL query returned no results or had issues.
        User Query: {user_query}
        Currently selected tables: {selected_block if selected_block else "(none)"}

        Here are tables related by foreign key or logical relationships to the selected tables:
        {candidates_block}

        From this list, choose ONLY the tables that are likely to help answer the user query.
        Prefer tables that:
        - Directly store the entities mentioned in the query
        - Or are clearly needed to join those entities

        Return ONLY valid JSON in this format:
        {{"selected_related_tables": ["table1", "table2", ...], "reasoning": "short explanation"}}
    """
    try:
        resp = await asyncio.wait_for(
            orchestrator.json_llm.ainvoke([HumanMessage(content=prompt)]),
            timeout=60.0
        )
        content = resp.content.strip()
        if content.startswith("```"):
            parts = content.split("```")
            if len(parts) >= 2:
                content = parts[1].strip()
        if content.lower().startswith("json"):
            content = content[4:].strip()
        data = json.loads(content)
        chosen = data.get("selected_related_tables", [])
        chosen = [t for t in chosen if t in candidate_tables]
        chosen = chosen[:3]

        if not chosen:
            logger.debug("[relationship_manager] LLM did not select any useful related tables")
            state["related_tables"] = []
            return state

        logger.info("[relationship_manager] LLM-selected related tables: %s", ', '.join(chosen))
        state["related_tables"] = chosen
        state["selected_tables"] = selected_tables + chosen
        state["retry_count"] = state.get("retry_count", 0) + 1
        state["messages"].append(AIMessage(content=f"Expanded to related tables: {', '.join(chosen)}"))

    except Exception as e:
        logger.warning("[relationship_manager] error: %s", e)
        fallback = candidate_tables[:2]
        logger.warning("[relationship_manager] using fallback tables: %s", ', '.join(fallback))
        state["related_tables"] = fallback
        state["selected_tables"] = selected_tables + fallback
        state["retry_count"] = state.get("retry_count", 0) + 1
        state["messages"].append(AIMessage(content=f"Expanded to fallback related tables: {', '.join(fallback)}"))

    return state


import requests

async def natural_language_response_agent(orchestrator, state: AgentState) -> AgentState:
    """Convert SQL results into a human-readable answer."""
    logger.debug("[nlr_agent] generating human-readable answer...")
    
    # Fetch live exchange rates (fail gracefully)
    live_rates = {}
    try:
        resp = requests.get("https://api.exchangerate-api.com/v4/latest/USD", timeout=2)
        if resp.status_code == 200:
            live_rates = resp.json().get("rates", {})
    except Exception as e:
        logger.warning("[nlr_agent] Failed to fetch live exchange rates: %s", e)

    execution_result = state.get("execution_result", {})
    if not execution_result.get("success"):
        state["natural_language_answer"] = "I couldn't retrieve the data. Please try rephrasing your question."
        return state

    rows = execution_result.get("rows", [])
    row_count = execution_result.get("row_count", 0)
    user_query = state["user_query"].lower()
    history = state.get("history", [])
    history_text = _format_conversation_history(history)
    scope_mode = state.get("scope_mode", "single")
    tenant_id = state.get("tenant_id")
    allowed_partner_ids = state.get("allowed_partner_ids") or []
    org_names = state.get("org_names", {})
    min_date, max_date = await orchestrator.get_data_window()

    def _org_label(pid):
        return org_names.get(str(pid), org_names.get(pid, f"organization {pid}"))

    if scope_mode == "consolidated":
        scope_label = f"tenant {tenant_id} ({len(allowed_partner_ids)} organizations)"
    else:
        only = allowed_partner_ids[0] if allowed_partner_ids else None
        scope_label = _org_label(only) if only is not None else "your organization"

    if row_count == 0:
        generated_sql = state.get("generated_sql", "")
        answer = (
            f"No records found for {scope_label} matching your query. "
            f"Note: available data covers {min_date} to {max_date}. "
            f"Try rephrasing with a specific category (e.g. sales invoices, purchase invoices, "
            f"customer payments) or a date range within that window."
        )
        if generated_sql:
            logger.debug("[nlr_agent] generated SQL (no rows): %s", generated_sql)
    else:
        row_preview = rows[:50]
        consolidated_note = ""
        if scope_mode == "consolidated":
            org_map = ", ".join(f"{pid}={_org_label(pid)}" for pid in allowed_partner_ids)
            consolidated_note = (
                f"\n            Note: This query was executed in a multi-org environment. "
                f"If the user explicitly requested data for ONLY ONE specific organization in their query, your answer MUST clearly state that the result is ONLY for that specific organization. Do NOT claim the total is 'across all organizations' or 'across the tenant' if they asked for a specific one."
            )
        prompt = f"""
            Summarize the database results into a clear, neat answer for the user.
            
            Conversation History:
            {history_text}
            
            User Query: {user_query}
            Scope: {scope_label}
            Results (First {len(row_preview)} of {row_count}): {json.dumps(row_preview, default=str)}{consolidated_note}
            
            Live Exchange Rates (Base USD): {json.dumps(live_rates)}

            Rules:
            1. Be concise and professional.
            2. SECURITY RULE: NEVER mention technical IDs (like primary keys, tenant IDs, organization IDs) unless specifically asked. NEVER expose internal database table names or database schema details in your response. Speak purely in natural business terms.
            3. CRITICAL CURRENCY RULE: NEVER hallucinate random currency symbols (like Nigerian Naira ₦). If the SQL results do not explicitly provide a currency, ONLY use USD ($) or INR (₹), or simply output the raw number. If the SQL results DO contain a currency field, format the output using that specific currency.
            4. CURRENCY CONVERSION RULE: If the user asks to convert to a currency, perform the math yourself using the Live Exchange Rates provided above. ALWAYS display the original base value with its unit first, followed by the converted value (e.g., "$100 USD (~ ₹9,500 INR)"). 
            5. If the user explicitly provided their own exchange rate in the prompt, USE THEIR RATE instead of the live rate. 
            6. If the requested currency is not in the DB and not in the live rates, state that it's unavailable and ask them to provide an exchange rate.
            7. Return ONLY the answer text.
            """
        try:
            resp = await asyncio.wait_for(
                orchestrator.sql_llm.ainvoke(
                    [HumanMessage(content=prompt)],
                    config={"tags": ["final_answer_llm"]}
                ),
                timeout=20.0
            )
            content = resp.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1].strip()
            if content.lower().startswith("json"):
                content = content[4:].strip()
            try:
                loaded = json.loads(content)
                if isinstance(loaded, dict) and "summary" in loaded:
                    answer = loaded["summary"]
                else:
                    answer = content
            except json.JSONDecodeError:
                answer = content
            answer = answer.strip('"').strip("'")
        except Exception as e:
            logger.warning("[nlr_agent] LLM summarization failed: %s", e)
            print(f"DEBUG EXCEPTION: {repr(e)}")
            answer = f"Found {row_count} records for {scope_label}. (Summary unavailable)"

    state["natural_language_answer"] = answer
    logger.debug("[nlr_agent] answer: %s", answer)
    return state


async def follow_up_agent(orchestrator, state: AgentState) -> AgentState:
    """Generate 3 contextual follow-up questions based on the answer and history."""
    logger.debug("[follow_up_agent] generating follow-up questions")
    
    # Only generate follow-ups if we had a successful answer and no clarification
    if state.get("clarification_needed") or not state.get("execution_result", {}).get("success") and not state.get("kpi_fastpath_hit"):
        state["follow_up_questions"] = []
        return state

    user_query = state.get("user_query", "")
    answer = state.get("natural_language_answer", "")
    
    prompt = f"""
    You are an intelligent financial AI assistant. Based on the user's last question and your answer, suggest exactly 3 relevant follow-up questions the user might want to ask next.
    
    The follow-up questions should:
    1. Be highly contextual to the current topic.
    2. Suggest drilling down (e.g., "Break this down by month", "Which customer contributed the most?").
    3. Suggest comparing to another period (e.g., "How does this compare to last year?").
    4. Be short, natural, and directly actionable.
    
    User Query: {user_query}
    Your Answer: {answer}
    
    Return the response strictly as a JSON object with a single key "questions" containing a list of 3 strings.
    Example: {{"questions": ["What is the breakdown by organization?", "How does this compare to last quarter?", "Who are the top 5 customers?"]}}
    """
    
    try:
        resp = await asyncio.wait_for(
            orchestrator.json_llm.ainvoke(
                [HumanMessage(content=prompt)],
                config={"tags": ["follow_up_llm"]}
            ),
            timeout=10.0
        )
        content = resp.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1].strip()
        if content.lower().startswith("json"):
            content = content[4:].strip()
            
        loaded = json.loads(content)
        questions = loaded.get("questions", [])
        if isinstance(questions, list):
            state["follow_up_questions"] = questions[:3]
        else:
            state["follow_up_questions"] = []
    except Exception as e:
        logger.warning("[follow_up_agent] failed to generate follow-ups: %s", e)
        state["follow_up_questions"] = []
        
    return state




# ---------------------------------------------------------------------------
# Compound query executor — runs each sub-query through the existing pipeline
# ---------------------------------------------------------------------------

async def compound_query_executor_agent(orchestrator, state: AgentState) -> AgentState:
    """Execute each sub-query through the existing single-query pipeline and collect results."""
    sub_query_plan = state["sub_query_plan"]
    sub_results = []

    for sq in sub_query_plan:
        logger.info("[compound_executor] running sub-query '%s': %s", sq["id"], sq["description"])

        sub_query_text = f"{sq['description']} (focus on {sq['focus']})"

        sub_state: AgentState = {
            "messages": [HumanMessage(content=sub_query_text)],
            "user_query": sub_query_text,
            "db_config": state["db_config"],
            "schema_metadata": state["schema_metadata"],   # reuse cached schema
            "selected_tables": [],
            "table_structures": {},
            "generated_sql": "",
            "validation_result": {},
            "execution_result": {},
            "retry_count": 0,
            "related_tables": [],
            "final_response": "",
            "error": "",
            "scope_mode": state["scope_mode"],
            "tenant_id": state["tenant_id"],
            "allowed_partner_ids": state["allowed_partner_ids"],
            "org_names": state["org_names"],
            "natural_language_answer": "",
            "intent": "finance_query",
            # Prevent recursive decomposition inside a sub-query
            "is_compound": False,
            "sub_query_plan": [],
            "sub_query_results": [],
            "kpi_fastpath_hit": False,
        }

        try:
            # Try KPI fast-path first; if it misses, run the full SQL pipeline
            sub_state = await kpi_fastpath_agent(orchestrator, sub_state)

            if not sub_state.get("kpi_fastpath_hit"):
                sub_state = await table_selection_agent(orchestrator, sub_state)
                sub_state = await structure_capture_agent(orchestrator, sub_state)
                sub_state = await sql_generation_agent(orchestrator, sub_state)
                sub_state = await review_agent(orchestrator, sub_state)
                sub_state = await execution_agent(orchestrator, sub_state)

            exec_result = sub_state.get("execution_result", {})
            sub_results.append({
                "id": sq["id"],
                "description": sq["description"],
                "sql": sub_state.get("generated_sql", ""),
                "rows": exec_result.get("rows", []),
                "row_count": exec_result.get("row_count", 0),
                "columns": exec_result.get("columns", []),
                "success": exec_result.get("success", False),
                "source": exec_result.get("source", "sql"),
            })
            logger.info(
                "[compound_executor] sub-query %s done: rows=%d",
                sq["id"], exec_result.get("row_count", 0),
            )

        except Exception as e:
            logger.error("[compound_executor] sub-query %s failed: %s", sq["id"], e)
            sub_results.append({
                "id": sq["id"],
                "description": sq["description"],
                "sql": "",
                "rows": [],
                "row_count": 0,
                "success": False,
                "error": str(e),
            })

    state["sub_query_results"] = sub_results
    state["messages"].append(AIMessage(content=f"Executed {len(sub_results)} sub-queries"))
    return state


# ---------------------------------------------------------------------------
# Result merger — synthesises sub-query results into a single NL answer
# ---------------------------------------------------------------------------

async def result_merger_agent(orchestrator, state: AgentState) -> AgentState:
    """Merge sub-query results and generate a unified natural-language answer."""
    sub_results = state["sub_query_results"]
    original_query = state["user_query"]

    if not any(sr["success"] for sr in sub_results):
        state["natural_language_answer"] = (
            "I couldn't retrieve the data for one or more parts of your query. "
            "Please try breaking it into simpler questions."
        )
        state["execution_result"] = {"success": False, "row_count": 0, "rows": []}
        return state

    results_context = [
        {
            "description": sr["description"],
            "row_count": sr["row_count"],
            "data": sr["rows"][:30],   # cap per sub-result to keep prompt size manageable
            "success": sr["success"],
        }
        for sr in sub_results
    ]
    all_sqls = [sr["sql"] for sr in sub_results if sr.get("sql")]

    prompt = f"""You are a finance data analyst. Multiple database queries were run to answer one user question.
Synthesize all results into a single, coherent, professional answer.

Original user question: {original_query}

Sub-query results:
{json.dumps(results_context, default=str, indent=2)}

Rules:
1. Combine the results meaningfully — don't just list them separately.
2. Highlight comparisons, totals, or insights where relevant.
3. Be concise and professional.
4. If any sub-query failed or returned no data, mention it briefly.
5. Do not mention SQL, sub-queries, or technical implementation details.
6. Return ONLY the answer text."""

    try:
        resp = await asyncio.wait_for(
            orchestrator.fast_llm.ainvoke(
                [HumanMessage(content=prompt)],
                config={"tags": ["final_answer_llm"]}
            ),
            timeout=30.0
        )
        answer = resp.content.strip().strip('"').strip("'")
    except Exception as e:
        logger.warning("[result_merger] LLM synthesis failed: %s", e)
        parts = [f"{sr['description']}: {sr['row_count']} record(s)" for sr in sub_results if sr["success"]]
        answer = "Here's what I found:\n" + "\n".join(parts)

    state["natural_language_answer"] = answer
    # Expose all SQLs in the generated_sql field so the UI "Show SQL" dropdown still works
    state["generated_sql"] = "\n\n-- Sub-query separator --\n\n".join(all_sqls)
    state["execution_result"] = {
        "success": True,
        "row_count": sum(sr["row_count"] for sr in sub_results),
        "rows": [],   # per-sub-query rows live in sub_query_results
    }
    logger.info(
        "[result_merger] merged %d sub-queries → total rows=%d",
        len(sub_results), state["execution_result"]["row_count"],
    )
    return state


def _build_steps(state: AgentState) -> list:
    """Build a human-readable step-by-step trace matching the architecture diagram."""
    steps = []

    # 1. Load Conversation Memory
    history = state.get("history", [])
    steps.append({
        "step": "1. Load Conversation Memory",
        "status": "done",
        "detail": f"Loaded {len(history)} previous turns."
    })

    intent = state.get("intent", "finance_query")
    clarification_needed = state.get("clarification_needed", False)

    # 2. Intent & Slot Extraction
    steps.append({
        "step": "2. Intent & Slot Extraction",
        "status": "done",
        "detail": f"Intent identified as '{intent}'."
    })

    # Scope Resolution (Context & Scope)
    scope_mode = state.get("scope_mode", "single")
    allowed = state.get("allowed_partner_ids") or []
    tenant_id = state.get("tenant_id")
    scope_detail = f"Consolidated view for tenant {tenant_id} — partner_ids {allowed}." if scope_mode == "consolidated" else f"Single organization — partner_id {allowed[0] if allowed else 'N/A'}."
    steps.append({
        "step": "Scope Resolution",
        "status": "done",
        "detail": scope_detail,
    })

    # 3. Requirement Completeness Check
    if intent in ("ambiguous", "off_topic") or clarification_needed:
        if clarification_needed:
            steps.append({
                "step": "3. Requirement Completeness Check",
                "status": "error",
                "detail": "Missing required slots/context."
            })
            steps.append({
                "step": "3A. Clarification Planner",
                "status": "info",
                "detail": "Planned targeted follow-up question."
            })
        else:
            steps.append({
                "step": "3. Requirement Completeness Check",
                "status": "error",
                "detail": f"Query rejected as {intent}."
            })
        return steps

    steps.append({
        "step": "3. Requirement Completeness Check",
        "status": "done",
        "detail": "All required slots present."
    })

    # 4. Schema Discovery
    schema_meta = state.get("schema_metadata", {})
    table_count = len(schema_meta.get("tables", {})) if schema_meta else 0
    steps.append({
        "step": "4. Schema Discovery",
        "status": "done" if table_count > 0 else "skipped",
        "detail": f"Loaded database schema ({table_count} tables available).",
    })

    # 5. Table Description
    steps.append({
        "step": "5. Table Description",
        "status": "done" if table_count > 0 else "skipped",
        "detail": "Generated descriptions for relevant tables."
    })

    # 6. Table Selection
    selected = state.get("selected_tables") or []
    steps.append({
        "step": "6. Table Selection",
        "status": "done" if selected else "skipped",
        "detail": ", ".join(selected) if selected else "No tables selected.",
    })

    # 7. Structure Capture
    table_structs = state.get("table_structures", {})
    steps.append({
        "step": "7. Structure Capture",
        "status": "done" if table_structs else "skipped",
        "detail": f"Fetched columns and keys for {len(table_structs)} tables."
    })

    # 8. SQL Generation & Planning
    generated_sql = state.get("generated_sql", "")
    steps.append({
        "step": "8. SQL Generation & Planning",
        "status": "done" if generated_sql else "skipped",
        "detail": generated_sql or "No SQL generated.",
        "code": bool(generated_sql),
    })

    # 9 & 10. Scope Guard & Validation
    validation = state.get("validation_result", {})
    if validation:
        steps.append({
            "step": "9. Scope Guard (Security Check)",
            "status": "done",
            "detail": "Validated SQL for tenant/partner scope and read-only operations."
        })
        valid = validation.get("valid", True)
        issues = validation.get("issues", [])
        steps.append({
            "step": "10. SQL Validation",
            "status": "done",
            "detail": "Static validation passed." if valid and not issues
            else f"Issues found and corrected: {', '.join(issues) if issues else 'auto-corrected'}.",
        })

    # 11. Execution
    exec_result = state.get("execution_result", {})
    if exec_result.get("success"):
        steps.append({
            "step": "11. Execution",
            "status": "done",
            "detail": f"Query executed successfully — {exec_result.get('row_count', 0)} row(s) returned.",
        })
    elif state.get("error"):
        steps.append({
            "step": "11. Execution",
            "status": "error",
            "detail": exec_result.get("error", state.get("error", "Execution failed.")),
        })

    retry_count = state.get("retry_count", 0)
    if retry_count:
        steps.append({
            "step": "Retries",
            "status": "info",
            "detail": f"{retry_count} retry attempt(s) were made."
        })

    # 12. Result Interpretation
    steps.append({
        "step": "12. Result Interpretation",
        "status": "done",
        "detail": "Formatted the result into a conversational response.",
    })

    return steps


def finalize_response(orchestrator, state: AgentState) -> AgentState:
    """Prepare the final JSON response."""
    logger.debug("[finalize_response] building final response")
    exec_result = state.get("execution_result", {})
    steps = _build_steps(state)

    if state.get("clarification_needed"):
        response = {
            "status": "clarification",
            "answer": state.get("clarification_question", "Can you please clarify?"),
            "sql": "",
            "row_count": 0,
            "steps": steps,
        }
    elif exec_result.get("success"):
        response = {
            "status": "success",
            "answer": state.get("natural_language_answer", "Here are your results."),
            "sql": state.get("generated_sql", ""),
            "row_count": exec_result.get("row_count", 0),
            "steps": steps,
            "follow_up_questions": state.get("follow_up_questions", []),
        }
    else:
        response = {
            "status": "error",
            "error": state.get("error", "Unknown error"),
            "answer": (
                state.get("natural_language_answer")
                or "Sorry, I couldn't process your request. Please try again."
            ),
            "sql": state.get("generated_sql", ""),
            "row_count": 0,
            "steps": steps,
        }

    state["final_response"] = json.dumps(response, indent=2, default=str)
    return state
