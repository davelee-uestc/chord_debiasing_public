#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
exec "$PYTHON" -u download_resources.py "$@"
