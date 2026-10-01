#!/usr/bin/env bash
# Host-side preflight for Lavix Vault. Fails fast with actionable messages.
# Mirrors the in-Compose "preflight" gate (docker-compose.yaml), which is the
# enforcing check at `docker compose up` time; this script is the early,
# human-readable version plus host-only checks (keys, Ollama, ports).
#
# Usage:
#   ./scripts/preflight.sh [--models]   # --models also verifies required Ollama models
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

FAIL=0
fail() { printf 'FAIL: %s\n' "$1"; FAIL=1; }
warn() { printf 'WARN: %s\n' "$1"; }
ok() { printf 'ok: %s\n' "$1"; }

# --- 0. stray .env -------------------------------------------------------
# Compose would silently apply a project .env; this stack uses none, so its
# presence means a stale file is lying around. Warn only.
if [[ -f .env ]]; then
  warn ".env exists in the project folder but this stack reads no .env file — remove it to avoid confusion"
fi

command -v docker >/dev/null 2>&1 || fail "docker is not installed (need Docker 24+)"
docker compose version >/dev/null 2>&1 || fail "docker compose v2 plugin is missing"
command -v openssl >/dev/null 2>&1 || fail "openssl is required (key generation, password checks)"
command -v python3 >/dev/null 2>&1 || fail "python3 is required (compose credential checks)"
python3 -c 'import yaml' 2>/dev/null || fail "python3 with PyYAML is required (pip install pyyaml)"

# --- 1. secrets + settings from docker-compose.yaml ----------------------
# NOTE: BaseLoader keeps every scalar a string. SafeLoader would coerce
# `off` to boolean False and break the RERANKER_MODE check below.

if [[ ! -f docker-compose.yaml ]]; then
  fail "docker-compose.yaml is missing"
else
  CREDS_JSON="$(python3 - docker-compose.yaml <<'PYEOF'
import sys
import yaml

document = yaml.load(open(sys.argv[1], encoding="utf-8"), Loader=yaml.BaseLoader)
settings = (document or {}).get("x-settings", {}) or {}
keys = [
    "DB_PASSWORD",
    "MINIO_ROOT_PASSWORD",
    "MINIO_APP_SECRET_KEY",
    "SEARXNG_SECRET",
    "SECRET_KEY",
    "AGENT_CAPABILITY_SECRET",
    "OLLAMA_URL",
    "RERANKER_MODE",
    "RERANKER_MODEL",
    "RERANKER_URL",
    "EMBEDDING_MODEL",
    "EMBEDDING_DIMS",
    "GRAPH_MEMORY",
    "NEO4J_PASSWORD",
]
for key in keys:
    print(f"{key}={settings.get(key, '')}")
PYEOF
)"
  declare -A CRED
  while IFS='=' read -r key value; do
    CRED["$key"]="$value"
  done <<< "$CREDS_JSON"

  check_secret() {
    local name="$1" value="${CRED[$1]:-}" min="$2"
    if [[ -z "$value" ]]; then
      fail "$name is empty — set it in docker-compose.yaml (EDIT THIS SECTION) or run ./scripts/gen-passwords.sh --write"
      return
    fi
    if [[ "$value" == "CHANGE_ME" || "$value" == *"CHANGE_ME"* ]]; then
      fail "$name still contains a CHANGE_ME placeholder — set it in docker-compose.yaml or run ./scripts/gen-passwords.sh --write"
      return
    fi
    if [[ "$value" =~ [^A-Za-z0-9_-] ]]; then
      fail "$name contains characters outside [A-Za-z0-9_-] — regenerate with ./scripts/gen-passwords.sh --write"
      return
    fi
    if (( ${#value} < min )); then
      fail "$name is too short (${#value} chars, need >= $min) — regenerate with ./scripts/gen-passwords.sh --write"
      return
    fi
  }

  check_secret DB_PASSWORD 16
  check_secret MINIO_ROOT_PASSWORD 16
  check_secret MINIO_APP_SECRET_KEY 16
  check_secret SEARXNG_SECRET 32
  check_secret SECRET_KEY 32
  check_secret AGENT_CAPABILITY_SECRET 32

  # Uniqueness across all secrets. Placeholders don't count: unset values
  # already fail above with their own message.
  declare -A SEEN
  for key in DB_PASSWORD MINIO_ROOT_PASSWORD MINIO_APP_SECRET_KEY SEARXNG_SECRET SECRET_KEY AGENT_CAPABILITY_SECRET; do
    value="${CRED[$key]:-}"
    [[ -n "$value" && "$value" != *"CHANGE_ME"* ]] || continue
    if [[ -n "${SEEN[$value]:-}" ]]; then
      fail "$key reuses the value of ${SEEN[$value]} — every secret must be unique (regenerate with ./scripts/gen-passwords.sh --write)"
    else
      SEEN["$value"]="$key"
    fi
  done
  _real() { [[ -n "${CRED[$1]:-}" && "${CRED[$1]}" != *"CHANGE_ME"* && -n "${CRED[$2]:-}" && "${CRED[$2]}" != *"CHANGE_ME"* ]]; }
  { _real SECRET_KEY AGENT_CAPABILITY_SECRET && [[ "${CRED[SECRET_KEY]}" == "${CRED[AGENT_CAPABILITY_SECRET]}" ]]; } && fail "SECRET_KEY and AGENT_CAPABILITY_SECRET must be independent"
  { _real MINIO_ROOT_PASSWORD MINIO_APP_SECRET_KEY && [[ "${CRED[MINIO_ROOT_PASSWORD]}" == "${CRED[MINIO_APP_SECRET_KEY]}" ]]; } && fail "MINIO_ROOT_PASSWORD and MINIO_APP_SECRET_KEY must differ"

  OLLAMA_URL="${CRED[OLLAMA_URL]:-}"
  if [[ -z "$OLLAMA_URL" || "$OLLAMA_URL" == "CHANGE_ME" ]]; then
    fail "OLLAMA_URL is not set — edit docker-compose.yaml (EDIT THIS SECTION) and set OLLAMA_URL to http://IP:PORT of your Ollama server"
    OLLAMA_URL=""
  elif [[ ! "$OLLAMA_URL" =~ ^https?://[^/]+:[0-9]+(/.*)?$ ]]; then
    fail "OLLAMA_URL must be http(s)://host:port (got '$OLLAMA_URL')"
    OLLAMA_URL=""
  else
    ok "OLLAMA_URL=$OLLAMA_URL"
  fi

  # Reranker wiring: mode must be known, model must be a baked preset,
  # external mode needs a valid URL. Reachability stays warn-only.
  RERANKER_MODE="${CRED[RERANKER_MODE]:-local}"
  RERANKER_MODEL="${CRED[RERANKER_MODEL]:-BAAI/bge-reranker-base}"
  RERANKER_URL="${CRED[RERANKER_URL]:-http://reranker:7997}"
  case "$RERANKER_MODE" in local|external|off) ok "RERANKER_MODE=$RERANKER_MODE" ;;
    *) fail "RERANKER_MODE must be local, external or off (got '$RERANKER_MODE')" ;;
  esac
  case "$RERANKER_MODEL" in
    BAAI/bge-reranker-base|cross-encoder/ms-marco-MiniLM-L-6-v2) ok "RERANKER_MODEL=$RERANKER_MODEL" ;;
    *) fail "RERANKER_MODEL must be a baked preset (got '$RERANKER_MODEL'). Supported presets: BAAI/bge-reranker-base, cross-encoder/ms-marco-MiniLM-L-6-v2" ;;
  esac
  if [[ "$RERANKER_MODE" == "external" ]]; then
    if [[ ! "$RERANKER_URL" =~ ^https?://[^/]+:[0-9]+(/.*)?$ ]]; then
      fail "RERANKER_URL must be http(s)://host:port when RERANKER_MODE=external (got '$RERANKER_URL')"
    elif ! curl -fsS -m 8 "$RERANKER_URL/health" -o /dev/null 2>/dev/null; then
      warn "reranker not reachable at $RERANKER_URL/health (mode=external)"
    else
      ok "reranker reachable at $RERANKER_URL"
    fi
  fi
  [[ "$FAIL" == "0" ]] && ok "settings block present and valid"
fi

# --- 1b. graph memory (runs by default; password always required) --------
GRAPH_MEMORY="${CRED[GRAPH_MEMORY]:-true}"
if [[ "$GRAPH_MEMORY" != "true" && "$GRAPH_MEMORY" != "false" ]]; then
  fail "GRAPH_MEMORY must be true or false (got '$GRAPH_MEMORY')"
fi
GRAPH_PW="${CRED[NEO4J_PASSWORD]:-}"
if [[ -z "$GRAPH_PW" || "$GRAPH_PW" == "CHANGE_ME" ]]; then
  fail "NEO4J_PASSWORD is still CHANGE_ME — set a real 16+ character password or run ./scripts/gen-passwords.sh --write"
elif [[ "$GRAPH_PW" =~ [^A-Za-z0-9_-] ]]; then
  fail "NEO4J_PASSWORD contains characters outside [A-Za-z0-9_-] — run ./scripts/gen-passwords.sh --write"
elif (( ${#GRAPH_PW} < 16 )); then
  fail "NEO4J_PASSWORD is too short (${#GRAPH_PW} chars, need >= 16) — run ./scripts/gen-passwords.sh --write"
else
  ok "graph memory on, NEO4J_PASSWORD set"
fi
if [[ "$GRAPH_MEMORY" != "true" ]]; then
  warn "GRAPH_MEMORY is false: Neo4j still starts unless scaled away (docker compose up -d --scale neo4j=0)"
fi

# --- 2. RSA keys (missing-key checks stay host-side) ----------------------
# A directory where a key file belongs is left over from an older run where
# a bind mount auto-created it: remove it and regenerate the keys.
for key in secrets/private_4096.pem secrets/public_4096.pem; do
  if [[ -d "$key" ]]; then
    fail "$key is a directory (left over from a bind mount) — run: rmdir $key && ./scripts/generate-keys.sh --force"
  elif [[ ! -s "$key" ]]; then
    fail "$key is missing — run ./scripts/generate-keys.sh"
  fi
done
# The private key must be readable by container uid 10001 (owner, o+r, or
# an ACL entry). Docker does not always remap secrets mounts to 0444.
if [[ -s secrets/private_4096.pem ]]; then
  readable=0
  [[ "$(stat -c %u secrets/private_4096.pem 2>/dev/null || stat -f %u secrets/private_4096.pem)" == "10001" ]] && readable=1
  [[ "$(stat -c %a secrets/private_4096.pem 2>/dev/null || stat -f %Lp secrets/private_4096.pem)" =~ [4567]$ ]] && readable=1
  if command -v getfacl >/dev/null 2>&1 && getfacl -p secrets/private_4096.pem 2>/dev/null | grep -q "^user:10001:r"; then
    readable=1
  fi
  if [[ "$readable" != "1" ]]; then
    fail "secrets/private_4096.pem is not readable by container uid 10001 — run: setfacl -m u:10001:r-- secrets/private_4096.pem (or sudo chown 10001 secrets/private_4096.pem)"
  else
    ok "RSA keypair present and container-readable (gitignored, never commit)"
  fi
fi

# --- 3. Ollama reachability ------------------------------------------------
# Pinned helper default first (the agent's rewrite/planner steps always need
# it), then the configured embedding model. Warn-only with exact commands.
PINNED_HELPER_MODEL="llama3.2:3b"
if [[ -n "${OLLAMA_URL:-}" ]]; then
  ollama_host="${OLLAMA_URL#http://}"
  ollama_host="${ollama_host#https://}"
  ollama_host="${ollama_host%%:*}"
  ollama_host="${ollama_host%%/*}"
  if ! getent hosts "$ollama_host" >/dev/null 2>&1; then
    warn "$ollama_host does not resolve from this machine — skipping the reachability probe (expected for Docker-only names like host.docker.internal; the containers resolve it via extra_hosts)"
  elif curl -fsS -m 8 "$OLLAMA_URL/api/tags" -o /tmp/lavix-preflight-tags.json 2>/dev/null; then
    ok "Ollama reachable at $OLLAMA_URL"
    for role_model in "HELPER_MODEL:${PINNED_HELPER_MODEL}" "EMBEDDING_MODEL:${CRED[EMBEDDING_MODEL]:-snowflake-arctic-embed2:cpu}"; do
      role="${role_model%%:*}"; model="${role_model#*:}"
      if grep -q "$model" /tmp/lavix-preflight-tags.json 2>/dev/null; then
        ok "model present ($role): $model"
      else
        warn "model not found on Ollama ($role): $model — run: ollama pull $model"
      fi
    done
    if [[ "${1:-}" == "--models" ]]; then
      for model in llama3.2 qwen2.5vl snowflake-arctic-embed2; do
        if grep -q "$model" /tmp/lavix-preflight-tags.json 2>/dev/null; then
          ok "model present: $model"
        else
          warn "model not found on Ollama (chat/embeddings will fail): $model — run: ollama pull $model"
        fi
      done
    fi
    rm -f /tmp/lavix-preflight-tags.json
  else
    fail "Ollama not reachable at $OLLAMA_URL (/api/tags) — start Ollama first"
  fi
fi

# --- 4. host ports (warn-only; read from the published ports) ---------------
mapfile -t PUBLISHED < <(grep -oE '"[0-9]+:[0-9]+"' docker-compose.yaml | tr -d '"' | cut -d: -f1 | sort -u)
for port in ${PUBLISHED[@]:-}; do
  if (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null; then
    exec 3>&- 3<&-
    warn "port $port is already in use (stop the other service or change the published port in docker-compose.yaml)"
  fi
done

if [[ "$FAIL" != "0" ]]; then
  printf '\nPreflight FAILED — fix the items above, then re-run ./scripts/preflight.sh\n' >&2
  exit 1
fi
printf '\nPreflight passed. Next: docker compose build && docker compose up -d\n'
