#!/usr/bin/env bash
# Run the Stellaria backend in --dev mode (port 8765, token "dev", CORS *) against a dev
# library in .dev/. Pass --fake to simulate the engine (no GPU/torch work).
#
#   scripts/dev-backend.sh [--fake] [extra backend args...]
#
# Python: $STELLARIA_PYTHON, else the active virtualenv, else python3. The interpreter needs
# backend/requirements-runtime.txt (plus torch for real engine work).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STELLARIA="$(dirname "$HERE")"
ENGINE="$STELLARIA/engine"
if [[ -n "${STELLARIA_PYTHON:-}" ]]; then
  PYTHON="$STELLARIA_PYTHON"
elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
  PYTHON="$VIRTUAL_ENV/bin/python"
else
  PYTHON="python3"
fi

args=()
for a in "$@"; do
  if [[ "$a" == "--fake" ]]; then args+=(--fake-engine); else args+=("$a"); fi
done

mkdir -p "$STELLARIA/.dev/library" "$STELLARIA/.dev/data"
cd "$STELLARIA/backend"
exec "$PYTHON" -m stellaria_backend --dev \
  --library "$STELLARIA/.dev/library" \
  --data "$STELLARIA/.dev/data" \
  --engine "$ENGINE" \
  --resources "$STELLARIA/resources/manifest.json" \
  ${args[@]+"${args[@]}"}
