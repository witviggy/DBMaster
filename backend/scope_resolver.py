"""
Scope resolution for multi-tenant / consolidated queries.

Two modes:
  * SINGLE       -> data is strictly limited to one organization (finance.partner_id).
  * CONSOLIDATED -> data spans every organization that belongs to a tenant_id.

The mapping (tenant_id -> [partner_ids], partner_id -> organization_name) is built
from ``public.organization`` where ``organization.id`` == ``finance.partner_id``.
Only clean/matched rows are used; orphan partner_ids and NULL tenants are ignored.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field


logger = logging.getLogger(__name__)


@dataclass
class ScopeContext:
    """Resolved, deterministic scope for a single request."""
    mode: str                                  # "single" | "consolidated"
    tenant_id: int | None
    allowed_partner_ids: list[int]
    org_names: dict[int, str] = field(default_factory=dict)

    def as_state(self) -> dict:
        return {
            "scope_mode": self.mode,
            "tenant_id": self.tenant_id,
            "allowed_partner_ids": list(self.allowed_partner_ids),
            "org_names": dict(self.org_names),
        }


class ScopeError(ValueError):
    """Raised when a request cannot be resolved to a valid, safe scope."""


class ScopeResolver:
    """Builds and caches the tenant/organization mapping from public.organization."""

    def __init__(self, pool):
        self.pool = pool
        self._tenant_to_partners: dict[int, list[int]] = {}
        self._partner_to_tenant: dict[int, int] = {}
        self._partner_to_name: dict[int, str] = {}
        self._loaded = False

    async def load(self) -> None:
        """Load the organization mapping into memory. Safe to call repeatedly."""
        self._tenant_to_partners = {}
        self._partner_to_tenant = {}
        self._partner_to_name = {}

        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    WITH active_partners AS (
                        SELECT partner_id FROM finance.kpi_base_metrics
                        UNION
                        SELECT partner_id FROM finance.sales_invoice
                        UNION
                        SELECT partner_id FROM finance.purchase_invoice
                        UNION
                        SELECT partner_id FROM finance.journal
                    )
                    SELECT id, organization_name, tenant_id
                    FROM public.organization
                    WHERE tenant_id IS NOT NULL
                      AND id IN (SELECT partner_id FROM active_partners)
                    """
                )
        except Exception as e:
            # public.organization may be absent in some dumps -> single mode still works.
            logger.exception("Could not load organization scope mapping")
            self._loaded = True
            return

        logger.info(
            "public.organization query returned %d raw rows (tenant_id IS NOT NULL)",
            len(rows),
        )
        if not rows:
            logger.warning(
                "public.organization returned 0 rows — consolidated queries will fail for ALL tenants. "
                "Check that the table is populated and the app is connected to the correct database."
            )

        for row in rows:
            org_id = row["id"]
            tenant_id = row["tenant_id"]
            name = row["organization_name"]
            if org_id is None or tenant_id is None:
                continue
            self._partner_to_tenant[org_id] = tenant_id
            self._partner_to_name[org_id] = name or f"Organization {org_id}"
            self._tenant_to_partners.setdefault(tenant_id, []).append(org_id)

        for tenant_id in self._tenant_to_partners:
            self._tenant_to_partners[tenant_id] = sorted(set(self._tenant_to_partners[tenant_id]))

        self._loaded = True
        if not self._tenant_to_partners:
            logger.warning(
                "Tenant mapping is empty after load — no valid (id, tenant_id) pairs found in "
                "public.organization. Consolidated queries will fail for every tenant."
            )
        logger.info(
            "Organization scope mapping loaded: organizations=%d tenants=%d "
            "tenant_to_organization_ids=%s organization_id_to_tenant_id=%s",
            len(self._partner_to_tenant),
            len(self._tenant_to_partners),
            dict(sorted(self._tenant_to_partners.items())),
            dict(sorted(self._partner_to_tenant.items())),
        )

    @property
    def loaded(self) -> bool:
        return self._loaded

    def _names_for(self, partner_ids: list[int]) -> dict[int, str]:
        return {pid: self._partner_to_name[pid] for pid in partner_ids if pid in self._partner_to_name}

    def resolve(
        self,
        consolidated: bool,
        tenant_id: int | None = None,
        organization_id: int | None = None,
    ) -> ScopeContext:
        """Resolve request inputs into a deterministic ScopeContext."""
        logger.info(
            "Resolving scope: consolidated=%s tenant_id=%r tenant_id_type=%s "
            "organization_id=%r organization_id_type=%s loaded=%s",
            consolidated,
            tenant_id,
            type(tenant_id).__name__,
            organization_id,
            type(organization_id).__name__,
            self._loaded,
        )
        if consolidated:
            if tenant_id is None:
                raise ScopeError("tenant_id is required when consolidated_view is true.")
            allowed = self._tenant_to_partners.get(tenant_id, [])
            if not allowed:
                logger.error(
                    "No organizations for consolidated tenant: tenant_id=%r "
                    "tenant_id_type=%s tenant_to_organization_ids=%s "
                    "organization_id_to_tenant_id=%s mapping_tenant_count=%d",
                    tenant_id,
                    type(tenant_id).__name__,
                    dict(sorted(self._tenant_to_partners.items())),
                    dict(sorted(self._partner_to_tenant.items())),
                    len(self._tenant_to_partners),
                )
                raise ScopeError(
                    f"No organizations found for tenant_id={tenant_id}. "
                    f"Cannot build a consolidated view."
                )
            return ScopeContext(
                mode="consolidated",
                tenant_id=tenant_id,
                allowed_partner_ids=allowed,
                org_names=self._names_for(allowed),
            )

        # SINGLE mode
        if organization_id is None:
            raise ScopeError(
                "organization_id (partner_id) is required for single-organization queries."
            )
            
        if organization_id not in self._partner_to_tenant:
            raise ScopeError(f"Organization ID {organization_id} does not exist.")
            
        resolved_tenant = self._partner_to_tenant[organization_id]
        if tenant_id is not None and resolved_tenant != tenant_id:
            raise ScopeError(f"Organization {organization_id} does not belong to Tenant {tenant_id}.")
            
        return ScopeContext(
            mode="single",
            tenant_id=resolved_tenant,
            allowed_partner_ids=[organization_id],
            org_names=self._names_for([organization_id]),
        )
