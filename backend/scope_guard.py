"""
Deterministic, parser-based enforcement of tenant/organization scope.

This is the real security boundary: regardless of what the LLM produced, a query
is only allowed to run if EVERY tenant-scoped finance table it touches is filtered
to a subset of the caller's allowed partner_ids. Uses sqlglot's AST (not regex) so
CTEs, sub-queries, aliases and joins are handled correctly.
"""
from __future__ import annotations

import sqlglot
from sqlglot import exp


class ScopeViolation(Exception):
    """Raised when a query cannot be proven safe against the allowed partner_ids."""


def _short_table_name(table: exp.Table) -> str:
    """Return the bare table name (without schema) for matching."""
    return (table.name or "").lower()


def _collect_partner_predicates(tree: exp.Expression, current_select: exp.Select | None = None):
    """
    Collect partner_id predicates for a specific SELECT scope.

    Returns:
        by_alias: dict[str, set[int]]  -> alias/table -> allowed literal values found
        bad_operator: bool             -> True if partner_id used with an unsafe operator
        non_integer: bool              -> True if a partner_id literal isn't an integer
    """
    by_alias: dict[str, set[int]] = {}
    bad_operator = False
    non_integer = False

    def _is_partner_col(node) -> bool:
        return isinstance(node, exp.Column) and (node.name or "").lower() == "partner_id"

    def _alias_of(col: exp.Column) -> str:
        return (col.table or "").lower()

    def _as_int(literal) -> int | None:
        if isinstance(literal, exp.Literal) and literal.is_int:
            return int(literal.name)
        # allow negative / numeric wrapped
        if isinstance(literal, exp.Neg):
            return None
        return None

    def _in_scope(node) -> bool:
        if current_select is None:
            return True
        return node.parent_select is current_select

    # Equality: partner_id = <int>
    for eq in tree.find_all(exp.EQ):
        if not _in_scope(eq):
            continue
        left, right = eq.left, eq.right
        # Skip column-to-column join conditions (e.g. si.partner_id = cp.partner_id)
        if isinstance(left, exp.Column) and isinstance(right, exp.Column):
            continue
        col = left if _is_partner_col(left) else (right if _is_partner_col(right) else None)
        if col is None:
            continue
        lit = right if col is left else left
        val = _as_int(lit)
        if val is None:
            non_integer = True
            continue
        by_alias.setdefault(_alias_of(col), set()).add(val)

    # IN list: partner_id IN (<int>, <int>, ...)
    for in_expr in tree.find_all(exp.In):
        if not _in_scope(in_expr):
            continue
        col = in_expr.this
        if not _is_partner_col(col):
            continue
        exprs = in_expr.expressions or []
        if not exprs:  # IN (subquery) -> cannot prove safe
            bad_operator = True
            continue
        for lit in exprs:
            val = _as_int(lit)
            if val is None:
                non_integer = True
                continue
            by_alias.setdefault(_alias_of(col), set()).add(val)

    # Unsafe operators on partner_id (!=, >, <, >=, <=, NOT IN)
    for op in tree.find_all(exp.NEQ, exp.GT, exp.LT, exp.GTE, exp.LTE):
        if not _in_scope(op):
            continue
        if _is_partner_col(op.left) or _is_partner_col(op.right):
            bad_operator = True
    for not_node in tree.find_all(exp.Not):
        if not _in_scope(not_node):
            continue
        inner = not_node.this
        if isinstance(inner, exp.In) and _is_partner_col(inner.this):
            bad_operator = True

    return by_alias, bad_operator, non_integer


def enforce_partner_scope(
    sql: str,
    allowed_partner_ids: list[int],
    tenant_scoped_tables: set[str],
) -> tuple[bool, str]:
    """
    Prove that ``sql`` is limited to ``allowed_partner_ids``.

    Args:
        sql: the SQL to validate.
        allowed_partner_ids: the only partner_ids this request may read.
        tenant_scoped_tables: bare table names (lowercase, no schema) that carry a
            partner_id column and therefore MUST be filtered.

    Returns:
        (True, "") if safe, else (False, "<reason>").
    """
    allowed = set(allowed_partner_ids)
    if not allowed:
        return False, "No allowed partner_ids resolved for this request."

    try:
        tree = sqlglot.parse_one(sql, dialect="postgres")
    except Exception as e:
        return False, f"Could not parse SQL for scope validation: {e}"

    if tree is None or tree.find(exp.Select) is None:
        return False, "Only SELECT queries are allowed."

    # Global check for bad operators and non-integers
    global_by_alias, bad_operator, non_integer = _collect_partner_predicates(tree, current_select=None)

    if bad_operator:
        return False, "partner_id must only be filtered with '=' or 'IN (list)'."
    if non_integer:
        return False, "partner_id filter must use integer literals."

    # Every value used anywhere must be within the allowed set.
    all_values = set().union(*global_by_alias.values()) if global_by_alias else set()
    outside = all_values - allowed
    if outside:
        return False, (
            f"Query references partner_id(s) {sorted(outside)} outside the allowed "
            f"scope {sorted(allowed)}."
        )

    # Per-Select scope coverage check.
    # Each SELECT node is an independent scope. We only check tables that belong
    # DIRECTLY to that SELECT node (not its subqueries, which are checked independently).
    any_scoped = False
    for select_node in tree.find_all(exp.Select):
        branch_sources: list[tuple[str, str]] = []
        for table in select_node.find_all(exp.Table):
            if table.parent_select is not select_node:
                continue
            short = _short_table_name(table)
            if short in tenant_scoped_tables:
                alias = table.alias_or_name.lower()
                branch_sources.append((alias, short))

        if not branch_sources:
            continue
        any_scoped = True

        # Collect predicates specifically in THIS select scope
        branch_by_alias, _, _ = _collect_partner_predicates(tree, current_select=select_node)

        branch_single = len(branch_sources) == 1
        bare_vals = branch_by_alias.get("", set())

        for alias, short in branch_sources:
            alias_vals = branch_by_alias.get(alias, set())
            # A bare (unqualified) partner_id filter is acceptable only when
            # there is exactly one scoped table in this scope (unambiguous).
            effective = alias_vals or (bare_vals if branch_single else set())
            if not effective:
                return False, (
                    f"Tenant-scoped table '{short}' (alias '{alias or short}') is not "
                    f"filtered by partner_id."
                )
            if not effective.issubset(allowed):
                return False, (
                    f"Table '{short}' filters partner_id {sorted(effective)} outside "
                    f"allowed scope {sorted(allowed)}."
                )

    if not any_scoped and not global_by_alias:
        return False, "Query has no partner_id filter."

    return True, ""
