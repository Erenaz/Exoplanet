#!/usr/bin/env bash
set -euo pipefail

case "${1:-}" in
    mlp|cnn) model="$1"; shift ;;
    *) printf 'Usage: %s {mlp|cnn} [--data-dir PATH] [--output-dir PATH]\n' "$0" >&2; exit 2 ;;
esac

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
exec "${PYTHON:-python3}" -m "script.train_${model}" "$@"
