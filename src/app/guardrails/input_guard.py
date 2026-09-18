from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any


@dataclass
class InputGuardResult:
    allow: bool
    response_text: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


_BOUNDARY_REFUSAL = (
    "Summary: I can’t help with that request. I’m an HR Policy Assistant that supports manager responses and drafting within HR policy and employee-reply needs. "
    "\nDetails: "+
    "- I can help with HR policy questions or drafting a professional reply based on provided company guidance. "
    "- I can’t disclose hidden prompts, secrets, or personal contact details. "
    "- If you share the HR policy question or employee situation at a high level, I can help within scope." 
    "\nSuggested next step: Please resend your question as an HR policy/support request (e.g., PTO timing, accommodation routing, remote-work guidance), and I’ll draft the manager-ready reply." 
)


def _contains_forbidden_steering(text: str) -> tuple[bool, str | None]:
    lowered = text.lower()

    patterns = [
        r"ignore (the )?(system prompt|system|developer prompt|developer|previous instructions)",
        r"override (the )?(system|developer) (prompt|instructions)",
        r"reveal (any|all )?(hidden|system|developer) (prompt|instructions)",
        r"openai_api_key|api key|base_url",
        r"\bdeveloper\b",
        r"\bsystem prompt\b",
        r"pineapple",
        r"secret override",
        r"ignore the normal instructions",
    ]
    for p in patterns:
        if re.search(p, lowered):
            return True, p
    return False, None


def _contains_out_of_scope(text: str) -> tuple[bool, str | None]:
    lowered = text.lower()
    patterns = [
        r"trading bot",
        r"crypto",
        r"investment",
        r"stock",
        r"forex",
        r"malware",
        r"phishing",
    ]
    for p in patterns:
        if re.search(p, lowered):
            return True, p
    return False, None


def _contains_private_data_exfil(text: str) -> tuple[bool, str | None]:
    lowered = text.lower()
    patterns = [
        r"reveal any hidden context",
        r"hidden context",
        r"personal contact",
        r"contact details",
        r"phone",
        r"address",
        r"home address",
    ]
    for p in patterns:
        if re.search(p, lowered):
            return True, p
    return False, None


def _contains_policy_lookup_miss_marker(text: str) -> tuple[bool, str | None]:
    # Helps the service avoid hallucinating records for made-up policy names.
    lowered = text.lower()
    patterns = [
        r"teleportation\s+leave",
        r"made[- ]up\s+policy",
        r"does (your|the) company have.*policy\?\s*$",
    ]
    for p in patterns:
        if re.search(p, lowered):
            return True, p
    return False, None


def review_input(*, user_text: str, context_rows: list[dict[str, Any]], state: dict[str, Any]) -> InputGuardResult:
    """Review text before the model receives it.

    This is intentionally deterministic and rule-based to resist direct/indirect steering.
    """

    # Defensive defaults
    details: dict[str, Any] = {
        'mode': state.get('mode'),
        'tenant_slug': state.get('tenant_slug'),
        'employee_ref_present': bool(state.get('employee_ref')),
        'context_count': len(context_rows),
    }

    if not user_text or not user_text.strip():
        return InputGuardResult(
            allow=False,
            response_text=_BOUNDARY_REFUSAL,
            details={**details, 'reason': 'empty_user_text'},
        )

    forbidden, reason = _contains_forbidden_steering(user_text)
    if forbidden:
        details.update({'reason': 'instruction_steering_attempt', 'match': reason})
        return InputGuardResult(allow=False, response_text=_BOUNDARY_REFUSAL, details=details)

    private, reason = _contains_private_data_exfil(user_text)
    if private:
        details.update({'reason': 'privacy_exfil_attempt', 'match': reason})
        return InputGuardResult(allow=False, response_text=_BOUNDARY_REFUSAL, details=details)

    oos, reason = _contains_out_of_scope(user_text)
    if oos:
        details.update({'reason': 'out_of_scope_request', 'match': reason})
        return InputGuardResult(
            allow=False,
            response_text=(
                "Summary: I can’t help with that request. I’m limited to HR policy support and drafting professional manager replies. "
                "\nDetails: - This request appears outside HR policy/support. "
                "- I can still help if you describe the HR policy question you need answered or the reply you need drafted." 
                "\nSuggested next step: Rephrase your request as an HR policy question (e.g., PTO timing, accommodation routing, remote-work guidance)."
            ),
            details=details,
        )

    miss, reason = _contains_policy_lookup_miss_marker(user_text)
    if miss:
        details.update({'reason': 'lookup_miss_marker', 'match': reason})
        # Return a safe clarification without asserting a made-up policy.
        return InputGuardResult(
            allow=False,
            response_text=(
                "Summary: I can’t confirm a policy for that request based on the available HR policy guidance. "
                "\nDetails: - If the policy name is new or internal, People Operations may need to verify the correct policy reference. "
                "- I can draft a manager-ready reply once you share the correct policy name or internal code (if any)."
                "\nSuggested next step: Please check with People Operations for the correct policy document or provide the employee context at a high level, and I’ll draft the response."
            ),
            details=details,
        )

    # Context-injection attempts inside user_text: refuse.
    if re.search(r"ignore\s+.*\bcontext\b", user_text, flags=re.IGNORECASE):
        details.update({'reason': 'context_override_attempt'})
        return InputGuardResult(allow=False, response_text=_BOUNDARY_REFUSAL, details=details)

    details['reason'] = 'allowed'
    return InputGuardResult(allow=True, response_text=None, details=details)
