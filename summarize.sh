#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
exec "$PYTHON" -u report_results.py "$@"
