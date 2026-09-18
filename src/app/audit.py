from __future__ import annotations

import json
import logging
from typing import Any
from uuid import UUID

from src.app.config import get_settings
from src.app.db import execute

logger = logging.getLogger(__name__)


def create_audit_record(
    conversation_id: UUID | str,
    tenant_id: int,
    endpoint_mode: str,
    manager_email: str,
    user_message: str,
    retrieved_context: list[dict[str, Any]],
) -> int:
    row = execute(
        '''
        insert into ai_message_audit (
            conversation_id,
            tenant_id,
            endpoint_mode,
            manager_email,
            user_message,
            retrieved_context,
            model_name
        ) values (%s, %s, %s, %s, %s, %s::jsonb, %s)
        returning id
        ''',
        (
            str(conversation_id),
            tenant_id,
            endpoint_mode,
            manager_email,
            _trim(user_message),
            json.dumps(_compact_context(retrieved_context)),
            get_settings().openai_model,
        ),
    )
    if not row:
        raise RuntimeError('Could not create audit record')
    audit_id = int(row['id'])
    logger.info(
        json.dumps(
            {
                'event': 'audit_record_created',
                'audit_id': audit_id,
                'conversation_id': str(conversation_id),
                'tenant_id': tenant_id,
                'endpoint_mode': endpoint_mode,
            }
        )
    )
    return audit_id


def update_audit_record(
    audit_id: int,
    raw_model_output: str | None = None,
    returned_output: str | None = None,
    guard_summary: dict[str, Any] | None = None,
    error_text: str | None = None,
) -> None:
    execute(
        '''
        update ai_message_audit
        set raw_model_output = coalesce(%s, raw_model_output),
            returned_output = coalesce(%s, returned_output),
            guard_summary = coalesce(%s::jsonb, guard_summary),
            error_text = coalesce(%s, error_text)
        where id = %s
        ''',
        (
            _trim(raw_model_output) if raw_model_output is not None else None,
            _trim(returned_output) if returned_output is not None else None,
            json.dumps(guard_summary) if guard_summary is not None else None,
            _trim(error_text) if error_text is not None else None,
            audit_id,
        ),
    )
    logger.info(
        json.dumps(
            {
                'event': 'audit_record_updated',
                'audit_id': audit_id,
                'has_returned_output': returned_output is not None,
                'has_raw_model_output': raw_model_output is not None,
                'has_guard_summary': guard_summary is not None,
                'has_error': error_text is not None,
            }
        )
    )


def _compact_context(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact = []
    for row in rows:
        compact.append(
            {
                'source_type': row.get('source_type'),
                'title': row.get('title'),
                'tenant_slug': row.get('tenant_slug'),
                'heading': row.get('heading'),
                'body_preview': _trim(str(row.get('body', '')), 260),
            }
        )
    return compact


def _trim(value: str | None, max_len: int = 2000) -> str | None:
    if value is None:
        return None
    return value[:max_len]
