"""
LLM provider helpers for the open-deepthink **library** API.

Build LangChain chat models for OpenRouter or local llama.cpp without the
optional web UI. Keys come from arguments, environment, or ``get_settings()``.

Example::

    from deepthink import create_llm, run_qnn

    llm = create_llm()  # OPENROUTER_API_KEY from env / .env
    result = await run_qnn(llm, "How do we break this deadlock?")
"""

from __future__ import annotations

from typing import Any, Literal

from deepthink.config import Settings, get_settings

ProviderName = Literal["openrouter", "llamacpp"]


def create_llm(
    provider: ProviderName | None = None,
    *,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    settings: Settings | None = None,
    **kwargs: Any,
):
    """
    Construct a LangChain chat model for OpenRouter or llama.cpp.

    Parameters
    ----------
    provider :
        ``"openrouter"`` or ``"llamacpp"``. Defaults to settings.default_provider.
    model :
        Model id (OpenRouter slug or local model name).
    api_key :
        OpenRouter key (or llama.cpp key). Falls back to settings / env.
    base_url :
        Override API base (OpenRouter or llama.cpp OpenAI-compatible URL).
    temperature, max_tokens :
        Generation defaults from settings when omitted.
    settings :
        Optional pre-loaded Settings instance.
    **kwargs :
        Extra keyword args forwarded to the underlying chat class.

    Returns
    -------
    LangChain chat model (``ChatOpenAI`` or ``ChatLlamaCpp``).
    """
    cfg = settings or get_settings()
    prov: str = (provider or cfg.default_provider or "openrouter").lower()
    temp = cfg.temperature if temperature is None else temperature
    max_tok = cfg.max_tokens if max_tokens is None else max_tokens

    if prov == "openrouter":
        from langchain_openai import ChatOpenAI

        key = api_key or cfg.resolved_api_key()
        if not key:
            raise ValueError(
                "OpenRouter API key required. Set OPENROUTER_API_KEY, pass "
                "api_key=..., or use provider='llamacpp'."
            )
        # Match app.py / LangChain OpenAI-compatible constructors.
        return ChatOpenAI(
            model=model or cfg.openrouter_model,
            openai_api_key=key,
            openai_api_base=base_url or cfg.openrouter_base_url,
            temperature=temp,
            **kwargs,
        )

    if prov == "llamacpp":
        from deepthink.models import ChatLlamaCpp

        url = cfg.normalize_llamacpp_url(base_url or cfg.llamacpp_base_url)
        return ChatLlamaCpp(
            base_url=url,
            api_key=api_key or cfg.llamacpp_api_key,
            model=model or cfg.llamacpp_model,
            temperature=temp,
            max_tokens=max_tok,
            **kwargs,
        )

    raise ValueError(f"Unknown provider {provider!r}. Use 'openrouter' or 'llamacpp'.")


def create_chat_model(**kwargs: Any):
    """Alias for :func:`create_llm`."""
    return create_llm(**kwargs)
