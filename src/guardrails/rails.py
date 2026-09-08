import logfire
import os
import threading

# Heavy ML imports are intentionally lazy (inside functions).
# Loading nemoguardrails/langchain_groq at module level triggers transformers,
# which calls importlib.metadata.packages_distributions() — a Python 3.12 bug
# that hangs indefinitely with large virtualenvs on macOS ARM.
from src.config.config import settings
from src.guardrails.colang_rules import COLANG_CONTENT, YAML_CONTENT, RAIL_INDICATORS

_rails = None  # type: ignore[assignment]
_classifier_llm = None  # type: ignore[assignment]
_init_lock = threading.Lock()  # guards lazy initialisation
_enable_nemo = os.getenv("ENABLE_NEMO_GUARDRAILS", "false").lower() in {"1", "true", "yes"}

# ── Off-topic classifier prompt ───────────────────────────────────────────────
# Deliberately tight: only Odoo-related questions pass. The classifier NEVER
# generates an answer — it only returns YES or NO.
_TOPIC_CLASSIFIER_PROMPT = """\
You are a strict topic gate for an Odoo ERP documentation assistant.
Your ONLY job is to decide whether the user's question is about Odoo or not.

Odoo-related topics include: Odoo modules (CRM, Sales, Inventory, Accounting, HR,
Manufacturing, POS, Purchase, Project, Discuss, Website, eCommerce, Helpdesk, etc.),
Odoo configuration, Odoo workflows, Odoo installation, Odoo upgrades, Odoo troubleshooting,
Odoo integrations, and Odoo best practices.

Greetings and introductory small-talk (e.g. "hello", "hi", "thanks") are also acceptable.

Respond with a single word only:
- YES  -> if the question is Odoo-related or is a greeting/small-talk
- NO   -> if the question is about anything else

User message: {message}
Answer:"""

_OFF_TOPIC_RESPONSE = (
    "I'm specialised exclusively in Odoo documentation and can only answer questions "
    "related to Odoo. I'm not able to help with that topic. "
    "Please ask me something about Odoo modules, configuration, workflows, or features."
)


def initialize_rails() -> None:
    """
    Build the topic classifier and, when explicitly enabled, NeMo LLMRails.

    Two-layer guard architecture:
      Layer 1 — LLM topic classifier  -> fast YES/NO off-topic gate
      Layer 2 — NeMo LLMRails         -> structural hard-block rules
                                         (jailbreak, PII, harmful, system-prompt)
    """
    from langchain_groq import ChatGroq

    global _rails, _classifier_llm

    guard_llm = ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model=settings.GROQ_MODEL,
        temperature=0
    )

    # Layer 1: standalone classifier (same fast model, no extra cost)
    _classifier_llm = guard_llm

    logfire.info(f"🛡️ Topic classifier initialised ({settings.GROQ_MODEL}).")

    # NeMo's import path can stall on macOS ARM/Python 3.12. Keep the API
    # responsive by making this optional; the topic classifier remains active.
    if _enable_nemo:
        from nemoguardrails import RailsConfig, LLMRails

        config = RailsConfig.from_content(
            colang_content=COLANG_CONTENT,
            yaml_content=YAML_CONTENT
        )
        _rails = LLMRails(config, llm=guard_llm)
        logfire.info(f"🛡️ NeMo Guardrails initialised ({settings.GROQ_MODEL}).")
    else:
        logfire.warning(
            "NeMo Guardrails is disabled. Set ENABLE_NEMO_GUARDRAILS=true "
            "only in an environment where NeMo imports successfully."
        )


def _is_odoo_related(message: str) -> bool:
    """
    Ask the LLM classifier whether the message is Odoo-related.
    Returns True (pass through) or False (off-topic -> block).
    """
    prompt = _TOPIC_CLASSIFIER_PROMPT.format(message=message)
    response = _classifier_llm.invoke(prompt)
    answer = (response.content if hasattr(response, "content") else str(response)).strip().upper()
    logfire.info(f"🔍 Topic classifier answer: '{answer}' | query='{message[:80]}'")
    # Accept YES / YES. / YES! — anything starting with YES passes
    return answer.startswith("YES")


def guard(message: str) -> tuple[bool, str | None]:
    """
    Run a user message through the two-layer guardrail gate.
    Lazily initialises rails on first call (avoids macOS ARM startup hang).

    Layer 1 — Topic classifier (LLM YES/NO):
        Fast, reliable off-topic detection that doesn't depend on fuzzy
        example matching. Anything not Odoo-related is blocked here.

    Layer 2 — NeMo LLMRails:
        Structural hard-block flows for jailbreak, PII, harmful content,
        and system-prompt leak attempts.

    Returns:
        (True,  rail_response) — a rail fired; return this response immediately,
                                skip the RAG pipeline entirely.
        (False, None)          — message is clean; proceed to LangGraph.
    """
    global _rails, _classifier_llm

    # ── Lazy init (thread-safe) ───────────────────────────────────────────────
    if _classifier_llm is None:
        with _init_lock:
            if _classifier_llm is None:  # double-checked locking
                initialize_rails()

    with logfire.span("🛡️ Guardrails Check"):

        # ── Layer 1: Topic classifier ─────────────────────────────────────────
        with logfire.span("🔍 Topic Classifier"):
            try:
                odoo_related = _is_odoo_related(message)
            except Exception as exc:
                logfire.warning(f"⚠️ Topic classifier failed ({exc}) — allowing through.")
                odoo_related = True  # fail open: let NeMo decide

        if not odoo_related:
            logfire.info(f"🛡️ Off-topic blocked by classifier | query='{message[:80]}'")
            return True, _OFF_TOPIC_RESPONSE

        # ── Layer 2: NeMo hard-block rules ────────────────────────────────────
        if _rails is None:
            return False, None

        with logfire.span("🛡️ NeMo Hard-Block Check"):
            try:
                result = _rails.generate(messages=[{"role": "user", "content": message}])
                content = (
                    result.get("content", "") if isinstance(result, dict) else str(result)
                )
                fired = any(indicator in content for indicator in RAIL_INDICATORS)

                if fired:
                    logfire.info(f"🛡️ NeMo rail fired | query='{message[:80]}'")
                    return True, content
            except Exception as exc:
                logfire.warning(f"⚠️ NeMo check failed ({exc}) — proceeding to RAG.")

        logfire.info("✅ Guardrails passed.")
        return False, None
