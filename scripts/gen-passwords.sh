#!/usr/bin/env bash
# Generate strong random secrets for docker-compose.yaml (EDIT THIS SECTION).
#
# Values are hex, so they only ever contain [A-Za-z0-9_-] and never break
# DATABASE URLs or shell handling. MinIO usernames are fixed literals
# (lavix-root / lavix-app) in the compose file, not secrets.
#
# Usage:
#   ./scripts/gen-passwords.sh                 # print values (paste them by hand)
#   ./scripts/gen-passwords.sh --write         # patch CHANGE_ME entries in place
#
# --write only replaces values that are still exactly CHANGE_ME: it never
# overwrites a value you already set, and it never touches OLLAMA_URL
# (set that to http://IP:PORT by hand). It never creates a .env file.
set -Eeuo pipefail

WRITE=0
for arg in "$@"; do
  case "$arg" in
    --write) WRITE=1 ;;
    -h|--help)
      printf 'Usage: %s [--write]\n' "$0"
      exit 0
      ;;
    *) printf 'Unknown argument: %s (try --help)\n' "$arg" >&2; exit 2 ;;
  esac
done

command -v openssl >/dev/null 2>&1 || {
  printf 'Required command is unavailable: openssl\n' >&2
  exit 1
}

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="$ROOT_DIR/docker-compose.yaml"

new_value() {
  case "$1" in
    DB_PASSWORD) openssl rand -hex 32 ;;
    MINIO_ROOT_PASSWORD) openssl rand -hex 32 ;;
    MINIO_APP_SECRET_KEY) openssl rand -hex 32 ;;
    SEARXNG_SECRET) openssl rand -hex 32 ;;
    SECRET_KEY) openssl rand -hex 48 ;;
    AGENT_CAPABILITY_SECRET) openssl rand -hex 48 ;;
    NEO4J_PASSWORD) openssl rand -hex 32 ;;
    *) printf 'Unknown credential: %s\n' "$1" >&2; exit 2 ;;
  esac
}

KEYS="DB_PASSWORD MINIO_ROOT_PASSWORD MINIO_APP_SECRET_KEY SEARXNG_SECRET SECRET_KEY AGENT_CAPABILITY_SECRET NEO4J_PASSWORD"

if [[ "$WRITE" != "1" ]]; then
  printf '# Paste these into the EDIT THIS SECTION block of docker-compose.yaml. Do NOT commit them.\n'
  for key in $KEYS; do
    printf '%s=%s\n' "$key" "$(new_value "$key")"
  done
  printf '# OLLAMA_URL is set by hand to http://IP:PORT (never generated).\n'
  exit 0
fi

patch_file() {
  local file="$1"; shift
  [[ -f "$file" ]] || {
    printf 'Compose file not found: %s\n' "$file" >&2
    exit 1
  }
  local patched=0 kept=0 key
  for key in "$@"; do
    # Only a definition line holding the bare token is patched
    # (e.g. "  DB_PASSWORD: ... CHANGE_ME"). Aliased consumers follow
    # automatically; user-edited values never match.
    if grep -Eq "^[[:space:]]*$key:[[:space:]]*.*[[:space:]]+CHANGE_ME([[:space:]]*(#.*)?)?$" "$file"; then
      value="$(new_value "$key")"
      sed -i -E "s/^([[:space:]]*$key:[[:space:]]*.*[[:space:]]+)CHANGE_ME([[:space:]]*(#.*)?)?$/\\1$value\\2/" "$file"
      printf 'patched %s in %s\n' "$key" "$(basename "$file")"
      patched=$((patched + 1))
    else
      printf 'kept %s in %s (already set by you)\n' "$key" "$(basename "$file")"
      kept=$((kept + 1))
    fi
  done
  printf 'done: %s patched, %s kept in %s. OLLAMA_URL left alone (set it to http://IP:PORT by hand).\n' "$patched" "$kept" "$(basename "$file")"
}

# shellcheck disable=SC2086
patch_file "$COMPOSE_FILE" $KEYS
printf 'Next: ./scripts/preflight.sh\n'
