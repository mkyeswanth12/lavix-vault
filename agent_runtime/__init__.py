"""Isolated, Ollama-only CUGA runtime for Lavix Vault.

The package initializer is intentionally dependency-free. Shared wire helpers
are imported by lean API and worker images that do not contain the complete
agent-runtime source tree; runtime configuration is imported explicitly from
``agent_runtime.config`` by the sidecar.
"""
