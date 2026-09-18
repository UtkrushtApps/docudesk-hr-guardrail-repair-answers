# Solution Steps

1. Implement rule-based guardrails in `src/app/guardrails/input_guard.py` to block (a) direct instruction override attempts, (b) retrieved-context steering prompts, (c) privacy exfil requests, and (d) clearly out-of-scope topics. Return a safe HR-scoped refusal/direction message when blocked.

2. Harden privacy and tenant boundaries in `src/app/retrieval.py` by restricting employee case lookups to the current tenant only and by returning only a privacy-minimized “case summary” to the model (no employee email/phone/address/notes). Also sanitize retrieved policy/procedure text to strip known prompt-injection/override fragments (e.g., “pineapple” style instructions).

3. Improve hallucination-resistance in `src/app/prompt_builder.py` by adding explicit guardrails to the constructed user prompt: use only provided context; if context is missing, ask clarifying questions and route to People Operations instead of inventing policy.

4. Integrate guardrails into the full request lifecycle in `src/app/pipeline.py`: after retrieval, run `review_input` before calling the LLM (and bypass the model on denial), then run `review_output` after the LLM to (a) redact sensitive patterns and (b) block prompt/secret leakage tokens. Ensure the same guard logic runs for both `/ask` and `/draft`, including follow-up turns, via a shared `state` object.

5. Add structured, auditable logging by enhancing `src/app/audit.py`: log JSON events when audit records are created/updated. In `pipeline.py`, pass a structured `guard_summary` into `update_audit_record`, including input/output guard decisions, context counts, and request metadata.

6. Run the service with the provided DB seed. If provider key is configured, run the invariant tests; confirm that (1) adversarial steering doesn’t appear in replies, (2) tenant-crossing employee data does not leak, and (3) private case details are redacted/never injected into model context.

