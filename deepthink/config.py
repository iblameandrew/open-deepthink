"""
Central configuration for open-deepthink.

Loads from (highest priority first):
  1. Environment variables (and process env set by the host)
  2. Optional config file (OPEN_DEEPTHINK_CONFIG or deepthink.toml / config.toml)
  3. .env file in the working directory or repo root
  4. Built-in defaults

API keys are never hard-coded. Prefer OPENROUTER_API_KEY (or API_KEY) in the
environment / .env. The web UI may still pass keys per-request via params;
those override settings for that request only and are not written to disk.
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _import_tomllib() -> Any | None:
    """Load stdlib tomllib (3.11+) or optional tomli; else None."""
    if sys.version_info >= (3, 11):
        import tomllib as mod

        return mod
    try:
        import tomli as mod

        return mod
    except ImportError:
        return None


tomllib = _import_tomllib()


ProviderName = Literal["openrouter", "llamacpp"]


def _repo_root() -> Path:
    """Return the repository root (parent of the deepthink package)."""
    return Path(__file__).resolve().parent.parent


def _find_env_file() -> str | None:
    """Locate a .env file without requiring one to exist."""
    candidates = [
        Path.cwd() / ".env",
        _repo_root() / ".env",
    ]
    for path in candidates:
        if path.is_file():
            return str(path)
    return None


def _load_toml_overrides() -> dict[str, Any]:
    """Optional flat/nested TOML config → settings field names."""
    env_path = os.environ.get("OPEN_DEEPTHINK_CONFIG", "").strip()
    candidates = []
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend(
        [
            Path.cwd() / "deepthink.toml",
            Path.cwd() / "config.toml",
            _repo_root() / "deepthink.toml",
            _repo_root() / "config.toml",
        ]
    )
    if tomllib is None:
        return {}
    for path in candidates:
        if not path.is_file():
            continue
        try:
            data = tomllib.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        flat: dict[str, Any] = {}
        for key, value in data.items():
            if isinstance(value, dict):
                for nested_key, nested_val in value.items():
                    # e.g. [openrouter] model → openrouter_model
                    flat[
                        f"{key}_{nested_key}" if not nested_key.startswith(key) else nested_key
                    ] = nested_val
                    # also accept exact field names inside sections
                    flat[nested_key] = nested_val
            else:
                flat[key] = value
        return flat
    return {}


class Settings(BaseSettings):
    """Typed application settings for server, providers, and QNN/QDAD defaults."""

    model_config = SettingsConfigDict(
        env_file=_find_env_file(),
        env_file_encoding="utf-8",
        env_prefix="",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Server ---
    host: str = Field(default="0.0.0.0", description="Bind host for the web server")
    port: int = Field(default=8000, ge=1, le=65535)
    log_level: str = Field(default="info")
    log_json: bool = Field(
        default=False,
        description="Emit structured JSON logs when true (Phase 2 full wiring)",
    )

    # --- Provider selection ---
    default_provider: ProviderName = Field(default="openrouter")
    openrouter_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("OPENROUTER_API_KEY", "openrouter_api_key"),
        description="OpenRouter API key (never commit this)",
    )
    api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("API_KEY", "api_key"),
        description="Fallback generic API key alias",
    )
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        validation_alias=AliasChoices("OPENROUTER_BASE_URL", "openrouter_base_url"),
    )
    openrouter_model: str = Field(
        default="stepfun/step-3.5-flash:free",
        validation_alias=AliasChoices("OPENROUTER_MODEL", "openrouter_model"),
    )
    openrouter_embedding_model: str = Field(
        default="google/gemini-embedding-001",
        validation_alias=AliasChoices("OPENROUTER_EMBEDDING_MODEL", "openrouter_embedding_model"),
    )

    llamacpp_base_url: str = Field(
        default="http://localhost:8080/v1",
        validation_alias=AliasChoices("LLAMACPP_BASE_URL", "llamacpp_base_url"),
    )
    llamacpp_api_key: str = Field(
        default="no-key-required",
        validation_alias=AliasChoices("LLAMACPP_API_KEY", "llamacpp_api_key"),
    )
    llamacpp_model: str = Field(
        default="llama-3.2-1b-instruct",
        validation_alias=AliasChoices("LLAMACPP_MODEL", "llamacpp_model"),
    )
    llamacpp_embedding_url: str = Field(
        default="http://localhost:8080/v1",
        validation_alias=AliasChoices("LLAMACPP_EMBEDDING_URL", "llamacpp_embedding_url"),
    )
    llamacpp_embedding_model: str = Field(
        default="text-embedding-nomic-embed-text-v1.5",
        validation_alias=AliasChoices("LLAMACPP_EMBEDDING_MODEL", "llamacpp_embedding_model"),
    )

    # --- LLM generation defaults ---
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1)
    request_timeout_s: float = Field(default=120.0, ge=1.0)
    max_retries: int = Field(default=3, ge=0, le=20)
    retry_backoff_base_s: float = Field(default=1.0, ge=0.0)

    # --- QNN (brainstorm) defaults ---
    qnn_mode: str = Field(default="auto")
    qnn_manual_layers: int = Field(default=3, ge=1)
    qnn_manual_width: int = Field(default=3, ge=1)
    qnn_num_epochs: int = Field(default=2, ge=1)
    qnn_vector_word_size: int = Field(default=6, ge=1)
    qnn_learning_rate: float = Field(default=0.5, ge=0.0, le=1.0)
    qnn_attention_top_k: int = Field(default=5, ge=0)
    qnn_enable_self_attention: bool = Field(default=True)

    # --- QDAD defaults ---
    qdad_grid_size: int = Field(default=3, ge=1, le=16)
    qdad_temperature_scale: float = Field(default=1.3, ge=0.0, le=2.0)
    qdad_denoising_steps: int = Field(default=2, ge=1, le=20)
    qdad_noun_verb_temperature: float = Field(default=0.6, ge=0.0, le=2.0)

    # --- Distillation defaults ---
    distillation_token_budget: int = Field(default=500_000, ge=1)
    distillation_output_dir: str = Field(default="distillation_output")

    # --- Paths / state ---
    open_deepthink_root: str | None = Field(
        default=None,
        validation_alias=AliasChoices("OPEN_DEEPTHINK_ROOT", "open_deepthink_root"),
    )
    state_dir: str = Field(
        default=".deepthink-state",
        description="Directory for optional run state / checkpoints",
    )
    skills_dir: str = Field(default="skills")

    # --- Safety / budgets (hooks for Phase 3) ---
    max_run_token_budget: int | None = Field(
        default=None,
        description="Optional hard cap on tokens per run (None = unlimited)",
    )
    enable_code_sandbox: bool = Field(default=True)

    @field_validator("default_provider", mode="before")
    @classmethod
    def _normalize_provider(cls, v: Any) -> str:
        if v is None or v == "":
            return "openrouter"
        s = str(v).strip().lower()
        if s in ("openrouter", "llamacpp"):
            return s
        raise ValueError("default_provider must be 'openrouter' or 'llamacpp'")

    def resolved_api_key(self) -> str | None:
        """Return the best available cloud API key without printing it."""
        return self.openrouter_api_key or self.api_key or None

    def normalize_llamacpp_url(self, url: str | None = None) -> str:
        """Normalize a llama.cpp OpenAI-compatible base URL to end with /v1."""
        base = (url or self.llamacpp_base_url or "").strip()
        base = base.rstrip("/")
        base = base.replace("/chat/completions", "")
        base = base.rstrip("/")
        if not base.endswith("/v1"):
            base = base + "/v1"
        return base

    def qnn_defaults(self) -> dict[str, Any]:
        """Params dict compatible with deepthink.qnn.default_qnn_params()."""
        return {
            "qnn_mode": self.qnn_mode,
            "manual_layers": self.qnn_manual_layers,
            "manual_width": self.qnn_manual_width,
            "num_epochs": self.qnn_num_epochs,
            "vector_word_size": self.qnn_vector_word_size,
            "learning_rate": self.qnn_learning_rate,
            "attention_top_k": self.qnn_attention_top_k,
            "enable_self_attention": self.qnn_enable_self_attention,
        }

    def qdad_defaults(self) -> dict[str, Any]:
        """Params dict compatible with QDAD clamp_params / pipeline."""
        return {
            "grid_size": self.qdad_grid_size,
            "n": self.qdad_grid_size,
            "temperature_scale": self.qdad_temperature_scale,
            "denoising_steps": self.qdad_denoising_steps,
            "noun_verb_temperature": self.qdad_noun_verb_temperature,
        }

    def provider_defaults(self) -> dict[str, Any]:
        """Common provider fields for UI/API payloads."""
        return {
            "provider": self.default_provider,
            "openrouter_model": self.openrouter_model,
            "llamacpp_url": self.normalize_llamacpp_url(),
            "llamacpp_model": self.llamacpp_model,
            "llamacpp_embedding_url": self.normalize_llamacpp_url(self.llamacpp_embedding_url),
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a process-wide Settings instance (cached).

    TOML file values fill gaps; environment variables and .env still win for
    fields that pydantic-settings reads from the environment.
    """
    overrides = _load_toml_overrides()
    # Only pass keys that Settings actually knows about
    known = set(Settings.model_fields.keys())
    filtered = {k: v for k, v in overrides.items() if k in known}
    return Settings(**filtered)


def reload_settings() -> Settings:
    """Clear the cache and reload (useful in tests)."""
    get_settings.cache_clear()
    return get_settings()
