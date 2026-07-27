"""Runtime configuration for the system-level CriderGPT Engine service."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    return default if value is None else value.strip().lower() in {"1", "true", "yes", "on"}


def _csv(name: str) -> tuple[str, ...]:
    return tuple(v.strip() for v in os.getenv(name, "").split(",") if v.strip())


@dataclass(frozen=True)
class Settings:
    host: str = os.getenv("ENGINE_HOST", "0.0.0.0")
    port: int = _int("ENGINE_PORT", 8000)
    log_level: str = os.getenv("LOG_LEVEL", "INFO").upper()
    api_keys: tuple[str, ...] = (
        _csv("CRIDERGPT_ENGINE_API_KEYS")
        or _csv("ENGINE_API_KEYS")
        or _csv("CRIDERGPT_ENGINE_API_KEY")
        or _csv("ENGINE_API_KEY")
    )

    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    ollama_vision_model: str = os.getenv("OLLAMA_VISION_MODEL", "llava:7b")
    ollama_timeout_seconds: int = _int("OLLAMA_TIMEOUT_SECONDS", 120)
    default_temperature: float = _float("OLLAMA_TEMPERATURE", 0.7)
    default_max_tokens: int = _int("OLLAMA_MAX_TOKENS", 2000)

    embedding_model: str = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
    embedding_timeout_seconds: int = _int("EMBEDDING_TIMEOUT_SECONDS", 30)
    rag_context_limit: int = _int("RAG_CONTEXT_LIMIT", 6)
    rag_similarity_threshold: float = _float("RAG_SIMILARITY_THRESHOLD", 0.55)
    rag_embedding_input_chars: int = _int("RAG_EMBEDDING_INPUT_CHARS", 8000)
    rag_chunk_prompt_chars: int = _int("RAG_CHUNK_PROMPT_CHARS", 1800)
    rag_total_prompt_chars: int = _int("RAG_TOTAL_PROMPT_CHARS", 7000)

    supabase_url: str | None = os.getenv("SUPABASE_URL")
    supabase_service_role_key: str | None = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    founder_name: str = os.getenv("CRIDERGPT_FOUNDER_NAME", "Jessie Crider")
    founder_email: str = os.getenv("CRIDERGPT_FOUNDER_EMAIL", "jessiecrider3@gmail.com")
    memory_limit: int = _int("MEMORY_CONTEXT_LIMIT", 20)
    writing_sample_limit: int = _int("WRITING_SAMPLE_LIMIT", 8)
    training_limit: int = _int("TRAINING_CONTEXT_LIMIT", 10)
    persist_direct_chat: bool = _bool("PERSIST_DIRECT_CHAT", False)

    image_backend: str = os.getenv("IMAGE_BACKEND", "automatic1111").lower()
    image_api_url: str = os.getenv("IMAGE_API_URL", "http://127.0.0.1:7860").rstrip("/")
    image_model: str | None = os.getenv("IMAGE_MODEL")
    image_timeout_seconds: int = _int("IMAGE_TIMEOUT_SECONDS", 180)
    max_reference_images: int = _int("MAX_REFERENCE_IMAGES", 4)
    allowed_reference_hosts: tuple[str, ...] = _csv("ALLOWED_REFERENCE_HOSTS")


settings = Settings()
