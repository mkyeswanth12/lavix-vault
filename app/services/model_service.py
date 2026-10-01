"""Ollama model discovery for the local-only chat deployment."""

from __future__ import annotations

import logging
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)

_EMBEDDING_FAMILIES = frozenset({"bert", "nomic-bert", "embed"})


class ModelService:
    """Discover local models and provide explicit provider client calls."""

    def __init__(self, ollama_base_url: str) -> None:
        self._base_url = ollama_base_url.rstrip("/")
        self._models_cache: list[dict[str, Any]] = []
        self._models_fetched_at = 0.0
        self._service_health_cache: dict[str, tuple[float, bool]] = {}
        self._model_capabilities_cache: dict[str, tuple[float, frozenset[str]]] = {}

    def get_available_models(self) -> list[dict[str, Any]]:
        """Return model metadata from Ollama's tags endpoint."""
        now = time.monotonic()
        if self._models_cache and now - self._models_fetched_at < 60:
            return list(self._models_cache)
        try:
            response = requests.get(f"{self._base_url}/api/tags", timeout=5)
            response.raise_for_status()
            models = []
            for raw_model in response.json().get("models", []):
                details = raw_model.get("details", {})
                models.append(
                    {
                        "name": raw_model.get("name", ""),
                        "size": raw_model.get("size", 0),
                        "digest": raw_model.get("digest", ""),
                        "families": details.get("families", []) or [],
                        "modalities": details.get("modalities", []) or [],
                        "parameter_size": details.get("parameter_size", ""),
                        "quantization_level": details.get("quantization_level", ""),
                    }
                )
        except (TypeError, ValueError, requests.RequestException):
            logger.info("Ollama model discovery unavailable", exc_info=True)
            return list(self._models_cache)

        self._models_cache = models
        self._models_fetched_at = now
        return list(models)

    def get_model_capabilities(self, model_name: str) -> frozenset[str]:
        """Return Ollama's declared capabilities for one installed model.

        Vision is validated from ``/api/show`` instead of guessing from a tag
        name.  Results are short-lived so an admin cannot save a stale model
        after it has been replaced locally.
        """

        normalized = model_name.strip()
        if not normalized:
            return frozenset()
        now = time.monotonic()
        cached = self._model_capabilities_cache.get(normalized)
        if cached is not None and now - cached[0] < 60:
            return cached[1]
        try:
            response = requests.post(
                f"{self._base_url}/api/show",
                json={"model": normalized},
                timeout=5,
            )
            response.raise_for_status()
            payload = response.json()
            capabilities = frozenset(
                str(value).strip().lower()
                for value in payload.get("capabilities", [])
                if str(value).strip()
            )
        except (TypeError, ValueError, requests.RequestException):
            logger.info("Ollama capability discovery unavailable for %s", normalized, exc_info=True)
            return frozenset()
        self._model_capabilities_cache[normalized] = (now, capabilities)
        return capabilities

    def is_service_available(self, url: str) -> bool:
        """Return a short-lived, bounded readiness result for a configured service."""

        target = url.strip()
        if not target:
            return False
        now = time.monotonic()
        cached = self._service_health_cache.get(target)
        if cached is not None and now - cached[0] < 30:
            return cached[1]
        try:
            response = requests.get(target, timeout=2)
            available = response.status_code < 400
        except requests.RequestException:
            available = False
        self._service_health_cache[target] = (now, available)
        return available

_service: ModelService | None = None


def init_model_service(ollama_base_url: str) -> ModelService:
    global _service
    _service = ModelService(ollama_base_url)
    return _service


def get_model_service() -> ModelService:
    if _service is None:
        raise RuntimeError("model service has not been initialised")
    return _service
