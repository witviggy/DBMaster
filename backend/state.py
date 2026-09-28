from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
import operator


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    user_query: str
    db_config: dict
    schema_metadata: dict
    selected_tables: list
    table_structures: dict
    generated_sql: str
    validation_result: dict
    execution_result: dict
    retry_count: int
    related_tables: list
    final_response: str
    error: str
    scope_mode: str
    tenant_id: int
    allowed_partner_ids: list
    org_names: dict
    natural_language_answer: str
    intent: str
    kpi_fastpath_hit: bool
    # Conversational state
    history: list
    clarification_needed: bool
    clarification_question: str
    # Compound query decomposition
    is_compound: bool
    sub_query_plan: list
    sub_query_results: list
    # Follow-up conversational AI
    follow_up_questions: list
