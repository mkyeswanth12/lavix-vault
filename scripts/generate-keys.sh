#!/usr/bin/env bash
# Generate the RSA-4096 signing keypair for Lavix Vault.
#
# Writes ./secrets/private_4096.pem (0600) and ./secrets/public_4096.pem
# (0644) in the repository root. Both paths are gitignored and must never
# be committed. The compose stack consumes them via the `secrets:` driver
# (private key) and a read-only bind mount (public key).
#
# Works on Linux and macOS (requires: openssl). Never overwrites existing
# keys unless --force is given.
#
# Usage:
#   ./scripts/generate-keys.sh [--force]
set -Eeuo pipefail

FORCE=0
case "${1:-}" in
  --force) FORCE=1 ;;
  "" ) ;;
  -h|--help)
    printf 'Usage: %s [--force]\n' "$0"
    exit 0
    ;;
  *) printf 'Unknown argument: %s (try --help)\n' "$1" >&2; exit 2 ;;
esac

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SECRETS_DIR="$ROOT_DIR/secrets"
PRIVATE_KEY="$SECRETS_DIR/private_4096.pem"
PUBLIC_KEY="$SECRETS_DIR/public_4096.pem"

command -v openssl >/dev/null 2>&1 || {
  printf 'Required command is unavailable: openssl\n' >&2
  exit 1
}

install -d -m 0700 "$SECRETS_DIR"

# A directory where a key file belongs is left over from an older run where
# a bind mount auto-created it: refuse with the exact fix, exit non-zero.
for existing in "$PRIVATE_KEY" "$PUBLIC_KEY"; do
  if [[ -d "$existing" ]]; then
    printf 'Found a directory at %s (left over from a bind mount).\n' "$existing" >&2
    printf 'Fix: rmdir %s && %s --force\n' "$existing" "$0" >&2
    exit 1
  fi
done

if [[ -s "$PRIVATE_KEY" && -s "$PUBLIC_KEY" && "$FORCE" != "1" ]]; then
  printf 'Keys already exist in %s (use --force to regenerate).\n' "$SECRETS_DIR"
  printf 'Next steps:\n'
  printf '  1. ./scripts/gen-passwords.sh --write   # fill credentials in docker-compose.yaml\n'
  printf '  2. edit docker-compose.yaml: set OLLAMA_URL to http://IP:PORT\n'
  printf '  3. ./scripts/preflight.sh\n'
  printf '  4. docker compose build && docker compose up -d\n'
  exit 0
fi

printf 'Generating RSA-4096 keypair in %s ...\n' "$SECRETS_DIR"
openssl genpkey -algorithm RSA -pkeyopt rsa_keygen_bits:4096 \
  -out "$PRIVATE_KEY" 2>/dev/null
openssl pkey -in "$PRIVATE_KEY" -pubout -out "$PUBLIC_KEY" 2>/dev/null
# The API/worker containers read the private key as uid 10001. Grant that
# uid read access without opening the file to anyone else: best-effort
# chown first (works as root), then a file ACL (works unprivileged),
# otherwise preflight will fail with the manual fix.
chown 10001:10001 "$PRIVATE_KEY" 2>/dev/null || true
chmod 0600 "$PRIVATE_KEY"
chmod 0644 "$PUBLIC_KEY"
if command -v setfacl >/dev/null 2>&1; then
  setfacl -m u:10001:r-- "$PRIVATE_KEY" 2>/dev/null || true
fi

printf 'Keys written (gitignored, never commit them).\n'
printf 'Back up %s somewhere safe: losing the private key makes encrypted documents unrecoverable.\n' "$PRIVATE_KEY"
printf 'Next steps:\n'
printf '  1. ./scripts/gen-passwords.sh --write   # fill credentials in docker-compose.yaml\n'
printf '  2. edit docker-compose.yaml: set OLLAMA_URL to http://IP:PORT\n'
printf '  3. ./scripts/preflight.sh\n'
printf '  4. docker compose build && docker compose up -d\n'
printf '\nRunning preflight to show the current state:\n'
"$ROOT_DIR/scripts/preflight.sh" || true
