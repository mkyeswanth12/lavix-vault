#!/bin/sh
# Lavix Vault preflight gate (runs as the `preflight` compose service).
# Fail-closed: exits non-zero naming the offending variable unless every
# secret is set, long enough, charset-clean and unique, OLLAMA_URL is a
# valid http(s)://host:port, and the reranker wiring is known. Services
# that need secrets depend on this gate, so nothing starts with CHANGE_ME.
# POSIX sh (busybox ash). Reads everything from the environment.
set -eu

fail() { echo "PREFLIGHT FAIL: $1" >&2; exit 1; }

# check_secret NAME VALUE MIN_LEN
check_secret() {
  name="$1"; value="$2"; min="$3"
  [ -n "$value" ] || fail "$name is empty — set it in docker-compose.yaml (EDIT THIS SECTION) or run ./scripts/gen-passwords.sh --write"
  [ "$value" != "CHANGE_ME" ] || fail "$name is still CHANGE_ME — set it in docker-compose.yaml (EDIT THIS SECTION) or run ./scripts/gen-passwords.sh --write"
  case "$value" in *CHANGE_ME*) fail "$name still contains a CHANGE_ME placeholder";; esac
  case "$value" in *[!A-Za-z0-9_-]*) fail "$name contains characters outside [A-Za-z0-9_-] — regenerate with ./scripts/gen-passwords.sh --write";; esac
  [ "${#value}" -ge "$min" ] || fail "$name is too short (${#value} chars, need >= $min) — regenerate with ./scripts/gen-passwords.sh --write"
}

check_secret DB_PASSWORD "$DB_PASSWORD" 16
check_secret MINIO_ROOT_PASSWORD "$MINIO_ROOT_PASSWORD" 16
check_secret MINIO_APP_SECRET_KEY "$MINIO_APP_SECRET_KEY" 16
check_secret SEARXNG_SECRET "$SEARXNG_SECRET" 32
check_secret SECRET_KEY "$SECRET_KEY" 32
check_secret AGENT_CAPABILITY_SECRET "$AGENT_CAPABILITY_SECRET" 32

# Graph memory runs by default: NEO4J_PASSWORD is always checked.
case "${GRAPH_MEMORY:-true}" in
  true|false) ;;
  *) fail "GRAPH_MEMORY must be true or false (got '$GRAPH_MEMORY')" ;;
esac
check_secret NEO4J_PASSWORD "$NEO4J_PASSWORD" 16

# Every secret must be unique (JWT != capability, root != app, ...).
seen=""; dups=""
for pair in "DB_PASSWORD:$DB_PASSWORD" "MINIO_ROOT_PASSWORD:$MINIO_ROOT_PASSWORD" "MINIO_APP_SECRET_KEY:$MINIO_APP_SECRET_KEY" "SEARXNG_SECRET:$SEARXNG_SECRET" "SECRET_KEY:$SECRET_KEY" "AGENT_CAPABILITY_SECRET:$AGENT_CAPABILITY_SECRET" "NEO4J_PASSWORD:$NEO4J_PASSWORD"; do
  n="${pair%%:*}"; v="${pair#*:}"
  case "$seen" in *"=$v;"*) dups="$dups $n";; *) seen="$seen$n=$v;";; esac
done
[ -z "$dups" ] || fail "duplicate secret values (each must be unique):$dups — regenerate with ./scripts/gen-passwords.sh --write"
[ "$SECRET_KEY" != "$AGENT_CAPABILITY_SECRET" ] || fail "SECRET_KEY and AGENT_CAPABILITY_SECRET must be independent"
[ "$MINIO_ROOT_PASSWORD" != "$MINIO_APP_SECRET_KEY" ] || fail "MINIO_ROOT_PASSWORD and MINIO_APP_SECRET_KEY must differ"

# OLLAMA_URL must be a valid http(s)://host:port.
[ -n "$OLLAMA_URL" ] || fail "OLLAMA_URL is empty — set it to http://IP:PORT of your Ollama server"
[ "$OLLAMA_URL" != "CHANGE_ME" ] || fail "OLLAMA_URL is still CHANGE_ME — set it to http://IP:PORT of your Ollama server"
case "$OLLAMA_URL" in http://*|https://*) ;; *) fail "OLLAMA_URL must start with http:// or https:// (got '$OLLAMA_URL')";; esac
hostport="${OLLAMA_URL#*://}"; hostport="${hostport%%/*}"
case "$hostport" in *:*) ;; *) fail "OLLAMA_URL must include a port (got '$OLLAMA_URL')";; esac
port="${hostport##*:}"
case "$port" in ''|*[!0-9]*) fail "OLLAMA_URL port must be numeric (got '$OLLAMA_URL')";; esac

# Reranker wiring: the mode must be known, the model must be a baked preset,
# and external mode needs a valid URL. Reachability stays warn-only: an
# external endpoint may be warming up.
[ -n "$RERANKER_MODE" ] || fail "RERANKER_MODE is empty — set local, external or off in docker-compose.yaml (x-settings)"
case "$RERANKER_MODE" in local|external|off) ;; *) fail "RERANKER_MODE must be local, external or off (got '$RERANKER_MODE')";; esac
case "$RERANKER_MODEL" in
  BAAI/bge-reranker-base|cross-encoder/ms-marco-MiniLM-L-6-v2) ;;
  *) fail "RERANKER_MODEL must be a baked preset (got '$RERANKER_MODEL'). Supported presets: BAAI/bge-reranker-base, cross-encoder/ms-marco-MiniLM-L-6-v2";;
esac
if [ "$RERANKER_MODE" = "external" ]; then
  [ -n "$RERANKER_URL" ] || fail "RERANKER_URL is empty — RERANKER_MODE=external needs an http(s)://host:port endpoint"
  case "$RERANKER_URL" in http://*|https://*) ;; *) fail "RERANKER_URL must start with http:// or https:// (got '$RERANKER_URL')";; esac
  urlhost="${RERANKER_URL#*://}"; urlhost="${urlhost%%/*}"
  case "$urlhost" in *:*) ;; *) fail "RERANKER_URL must include a port (got '$RERANKER_URL')";; esac
  urlport="${urlhost##*:}"
  case "$urlport" in ''|*[!0-9]*) fail "RERANKER_URL port must be numeric (got '$RERANKER_URL')";; esac
  wget -q -T 8 -O /dev/null "$RERANKER_URL/health" 2>/dev/null \
    || echo "PREFLIGHT WARN: reranker not reachable at $RERANKER_URL/health (mode=external)" >&2
fi

echo "preflight ok: secrets set, OLLAMA_URL=$OLLAMA_URL"
