#!/usr/bin/env bash
set -u
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)" || exit 1
exec python3 -u "$HERE/scripts/publish_jetson_results.py" --repo "$HERE" "$@"
