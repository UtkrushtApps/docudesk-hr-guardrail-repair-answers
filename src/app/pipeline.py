from __future__ import annotations

from typing import Any

from src.app.audit import create_audit_record, update_audit_record
from src.app.conversation import append_turn, get_or_create_conversation, get_recent_turns
from src.app.guardrails.input_guard import review_input
from src.app.guardrails.output_guard import review_output
from src.app.llm_client import ChatClient
from src.app.prompt_builder import build_messages
from src.app.retrieval import resolve_manager, resolve_tenant, retrieve_context, summarize_sources
from src.app.schemas import AskRequest, DraftRequest, ServiceResponse


def handle_docudesk_request(mode: str, request: AskRequest | DraftRequest) -> ServiceResponse:
    tenant = resolve_tenant(request.tenant_slug)
    if not tenant:
        raise RuntimeError('Unknown tenant')

    manager = resolve_manager(tenant['id'], request.manager_email)
    user_message = _message_from_request(request)
    employee_ref = getattr(request, 'employee_ref', None)

    conversation = get_or_create_conversation(
        tenant_id=tenant['id'],
        manager_email=request.manager_email,
        mode=mode,
        employee_ref=employee_ref,
        conversation_id=getattr(request, 'conversation_id', None),
    )

    # Persist user request turn
    append_turn(conversation['id'], 'user', user_message)

    prior_turns = get_recent_turns(conversation['id'], limit=6)
    context_rows = retrieve_context(tenant, manager, user_message, employee_ref)

    # Build shared state so guard decisions are consistent across turns.
    state = {
        'mode': mode,
        'tenant_slug': request.tenant_slug,
        'tenant_id': tenant['id'],
        'manager_email': request.manager_email,
        'manager_found': bool(manager),
        'employee_ref': employee_ref,
        'conversation_id': str(conversation['id']),
        'prior_turns_count': len(prior_turns),
        'context_count': len(context_rows),
    }

    audit_id = create_audit_record(
        conversation_id=conversation['id'],
        tenant_id=tenant['id'],
        endpoint_mode=mode,
        manager_email=request.manager_email,
        user_message=user_message,
        retrieved_context=context_rows,
    )

    messages = build_messages(
        mode=mode,
        user_message=user_message,
        context_rows=context_rows,
        prior_turns=prior_turns,
        tone=getattr(request, 'tone', None),
    )

    # Guard input BEFORE model
    input_guard = review_input(user_text=user_message, context_rows=context_rows, state=state)
    guard_summary: dict[str, Any] = {
        'input_guard': input_guard.details,
        'output_guard': {},
        'privacy': {
            'manager_found': state['manager_found'],
            'tenant_bounded_case_context': True,
            'context_count': len(context_rows),
        },
        'request_metadata': {
            'messages_count': len(messages),
            'user_message_chars': len(user_message),
            'prior_turns_count': len(prior_turns),
        },
    }

    if not input_guard.allow:
        final_reply = input_guard.response_text or (
            "Summary: I can’t help with that request. I’m limited to HR policy support and drafting professional manager replies. "
            "\nDetails: - Please rephrase as an HR policy/support request. "
            "\nSuggested next step: Provide the HR topic and the employee situation at a high level."
        )
        append_turn(conversation['id'], 'assistant', final_reply)
        update_audit_record(
            audit_id,
            raw_model_output=None,
            returned_output=final_reply,
            guard_summary=guard_summary,
        )
        return ServiceResponse(
            conversation_id=conversation['id'],
            audit_id=audit_id,
            reply=final_reply,
            sources=summarize_sources(context_rows),
            metadata={'mode': mode, 'context_count': len(context_rows), 'guarded': True},
        )

    raw_reply = ''
    try:
        raw_reply = ChatClient().complete(messages)
        output_guard = review_output(model_text=raw_reply, context_rows=context_rows, state=state)
        final_reply = output_guard.text

        guard_summary['output_guard'] = output_guard.details

        append_turn(conversation['id'], 'assistant', final_reply)
        update_audit_record(
            audit_id,
            raw_model_output=raw_reply,
            returned_output=final_reply,
            guard_summary=guard_summary,
        )

    except Exception as exc:
        guard_summary['error'] = {'text': str(exc)}
        update_audit_record(
            audit_id,
            raw_model_output=raw_reply or None,
            returned_output=None,
            guard_summary=guard_summary,
            error_text=str(exc),
        )
        raise

    return ServiceResponse(
        conversation_id=conversation['id'],
        audit_id=audit_id,
        reply=final_reply,
        sources=summarize_sources(context_rows),
        metadata={'mode': mode, 'context_count': len(context_rows), 'guarded': False},
    )


def _message_from_request(request: AskRequest | DraftRequest) -> str:
    if isinstance(request, AskRequest):
        return request.message
    return request.employee_message
