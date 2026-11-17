#!/usr/bin/env bash
set -u
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)" || exit 1
if ! command -v python3 >/dev/null 2>&1; then
    printf 'Python 3 bulunamadı. Bu başlatıcı Jetson Ubuntu içindir.\n' >&2
    exit 1
fi
exec python3 -u "$HERE/scripts/jetson_auto.py" --repo "$HERE" "$@"
