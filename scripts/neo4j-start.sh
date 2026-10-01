#!/bin/sh
# Neo4j start guard for Lavix Vault (mounted read-only into the neo4j service).
# Refuses a placeholder, empty, short, or odd-character password, then hands
# off to the stock Neo4j entrypoint. POSIX sh: runs as the container entrypoint.
set -eu

fail() { echo "neo4j: $1" >&2; exit 3; }

[ -n "${NEO4J_PASSWORD:-}" ] || fail "set a real NEO4J_PASSWORD (16+ characters, letters/digits/_/- only) in docker-compose.yaml"
[ "$NEO4J_PASSWORD" != "CHANGE_ME" ] || fail "NEO4J_PASSWORD is still CHANGE_ME — set a real password in docker-compose.yaml"
[ "${#NEO4J_PASSWORD}" -ge 16 ] || fail "NEO4J_PASSWORD is too short (${#NEO4J_PASSWORD} chars, need >= 16)"
case "$NEO4J_PASSWORD" in
  *[!A-Za-z0-9_-]*) fail "NEO4J_PASSWORD contains characters outside [A-Za-z0-9_-]" ;;
esac

export NEO4J_AUTH="neo4j/$NEO4J_PASSWORD"
unset NEO4J_PASSWORD
exec /startup/docker-entrypoint.sh neo4j
