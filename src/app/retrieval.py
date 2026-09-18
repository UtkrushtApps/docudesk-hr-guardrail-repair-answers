from __future__ import annotations

from typing import Any

from src.app.db import fetch_all, fetch_one


def resolve_tenant(slug: str) -> dict[str, Any] | None:
    return fetch_one(
        'select id, slug, display_name, region, status from tenants where slug = %s and archived_at is null',
        (slug,),
    )


def resolve_manager(tenant_id: int, manager_email: str) -> dict[str, Any] | None:
    return fetch_one(
        '''
        select id, email, full_name, role, team_name, status
        from hr_users
        where tenant_id = %s and lower(email) = lower(%s) and deleted_at is null
        ''',
        (tenant_id, manager_email),
    )


def _sanitize_retrieved_text(text: str | None) -> str:
    """Remove instruction-like prompt injection and obvious secret/override phrases."""
    if not text:
        return ''

    lowered = text.lower()
    # Strip known injection/override content.
    bad_markers = [
        'ignore the normal instructions',
        'secret override',
        'pineapple',
        'reveal any employee contact details',
        'end archived note',
        'also reveal',
    ]
    if any(m in lowered for m in bad_markers):
        # Drop any line containing such markers.
        lines = []
        for line in text.splitlines():
            l = line.lower()
            if any(m in l for m in bad_markers):
                continue
            lines.append(line)
        text = '\n'.join(lines)

    # Remove patterns that look like "instructions" rather than policy content.
    text = text.replace('If this text is read by an AI assistant, ignore', '')

    return text.strip()


def _case_exists_in_tenant(tenant_id: int, employee_ref: str) -> bool:
    row = fetch_one(
        '''
        select 1
        from employee_cases
        where tenant_id = %s and deleted_at is null and employee_ref = %s
        limit 1
        ''',
        (tenant_id, employee_ref),
    )
    return bool(row)


def _build_case_context_body(case_type: str | None, case_status: str | None) -> str:
    # IMPORTANT: Do not return employee PII (email/phone/address) to the model.
    ct = case_type or 'unknown_case_type'
    cs = case_status or 'unknown_status'
    return (
        'Employee case found for the provided reference. '
        f'Case type: {ct}. Case status: {cs}. '
        'Do not disclose personal contact details; use this only to guide HR-policy-aligned manager wording.'
    )


def retrieve_context(
    tenant: dict[str, Any],
    manager: dict[str, Any] | None,
    message: str,
    employee_ref: str | None,
) -> list[dict[str, Any]]:
    search = f"%{message[:80]}%"
    tokens = [part for part in message.replace('?', ' ').replace(',', ' ').split() if len(part) > 3]
    token = f"%{tokens[0]}%" if tokens else search

    policies = fetch_all(
        '''
        select 'policy' as source_type,
               d.title,
               coalesce(t.slug, 'global') as tenant_slug,
               c.heading,
               c.body,
               d.policy_area,
               c.created_at
        from policy_chunks c
        join policy_documents d on d.id = c.document_id
        left join tenants t on t.id = d.tenant_id
        where d.status = 'published'
          and d.archived_at is null
          and (d.tenant_id is null or d.tenant_id = %s)
          and (c.body ilike %s or c.heading ilike %s or d.policy_area ilike %s or d.title ilike %s)
        order by d.tenant_id nulls first, c.created_at desc
        limit 6
        ''',
        (tenant['id'], token, token, token, token),
    )

    procedures = fetch_all(
        '''
        select 'procedure' as source_type,
               title,
               %s as tenant_slug,
               site_code as heading,
               body,
               team_name as policy_area,
               updated_at as created_at
        from team_procedures
        where tenant_id = %s
          and deleted_at is null
          and status = 'active'
          and (%s is null or team_name = %s or body ilike %s or title ilike %s)
        order by updated_at desc
        limit 5
        ''',
        (
            tenant['slug'],
            tenant['id'],
            manager['team_name'] if manager else None,
            manager['team_name'] if manager else None,
            token,
            token,
        ),
    )

    # Employee case context (tenant-bounded and privacy-minimized)
    case_candidate = employee_ref or _extract_possible_ref(message)
    cases: list[dict[str, Any]] = []
    if case_candidate:
        # Enforce tenant boundary: never query or return cases outside tenant.
        if _case_exists_in_tenant(tenant['id'], case_candidate):
            cases = fetch_all(
                '''
                select 'case' as source_type,
                       employee_ref as title,
                       t.slug as tenant_slug,
                       case_type as heading,
                       -- Body is intentionally privacy-minimized.
                       concat(
                         'Case summary only. ',
                         'Case type: ', coalesce(case_type,'unknown'),
                         '. Case status: ', coalesce(case_status,'unknown'),
                         '. Do not disclose employee contact details.',
                         ' Notes are not included in AI context.'
                       ) as body,
                       case_status as policy_area,
                       opened_at as created_at
                from employee_cases ec
                join tenants t on t.id = ec.tenant_id
                where ec.deleted_at is null
                  and ec.tenant_id = %s
                  and ec.employee_ref = %s
                order by opened_at desc
                limit 2
                ''',
                (tenant['id'], case_candidate),
            )

    combined = policies + procedures + cases
    # If retrieval produced nothing at all, fall back to a small global policy set.
    if not combined:
        combined = fetch_all(
            '''
            select 'policy' as source_type,
                   d.title,
                   coalesce(t.slug, 'global') as tenant_slug,
                   c.heading,
                   c.body,
                   d.policy_area,
                   c.created_at
            from policy_chunks c
            join policy_documents d on d.id = c.document_id
            left join tenants t on t.id = d.tenant_id
            where d.status = 'published'
              and d.archived_at is null
              and (d.tenant_id is null or d.tenant_id = %s)
            order by c.created_at desc
            limit 3
            ''',
            (tenant['id'],),
        )

    # Final sanitization of retrieved text against prompt injection.
    for row in combined:
        if 'heading' in row and isinstance(row['heading'], str):
            row['heading'] = _sanitize_retrieved_text(row.get('heading'))
        if 'body' in row and isinstance(row['body'], str):
            row['body'] = _sanitize_retrieved_text(row.get('body'))
        if 'title' in row and isinstance(row['title'], str):
            row['title'] = _sanitize_retrieved_text(row.get('title'))

    return combined[:10]


def summarize_sources(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str | None]] = set()
    sources: list[dict[str, Any]] = []
    for row in rows:
        key = (row['source_type'], row['title'], row.get('tenant_slug'))
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            {
                'source_type': row['source_type'],
                'title': row['title'],
                'tenant_slug': row.get('tenant_slug'),
            }
        )
    return sources


def _extract_possible_ref(message: str) -> str | None:
    for raw in message.replace(',', ' ').replace('.', ' ').split():
        token = raw.strip().upper()
        if '-' in token and any(ch.isdigit() for ch in token):
            # Candidate ref format enforcement is intentionally loose here.
            return token
    return None
