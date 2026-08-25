#!/usr/bin/env bash
# Wrapper to forward command-line arguments to the unit-test executable.
# Resolve relative to this script so both `unit-tests/test.sh chiron-phs`
# and `cd unit-tests && ./test.sh chiron-phs` work.
set -euo pipefail
cd "$(dirname "$0")"
exec "./build/glades-unit-tests" "$@"
