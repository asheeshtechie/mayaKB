#!/bin/sh
set -e
# Apply database migrations before any command (idempotent; guarded by an advisory lock).
if [ -n "$MAYAKB_DATABASE_URL" ] && [ "${MAYAKB_SKIP_MIGRATIONS:-0}" != "1" ]; then
  mayakb db init >&2  # stderr, so command output (e.g. --json) stays clean
fi
exec "$@"
