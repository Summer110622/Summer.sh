#!/usr/bin/env bash
set -euo pipefail

ROOT="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"

if ! command -v python3 >/dev/null 2>&1; then
  printf 'ERROR: Python 3 is required.\n' >&2
  exit 127
fi

exec python3 "$ROOT/summer.py" "$@"
