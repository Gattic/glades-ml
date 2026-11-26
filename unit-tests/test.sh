#!/usr/bin/env bash
# Wrapper to forward command-line arguments to the unit-test executable.
# Resolve relative to this script so both `unit-tests/test.sh chiron-phs`
# and `cd unit-tests && ./test.sh chiron-phs` work.
#
# The documentation/benchmark consistency check
# (`unit-tests/checkdocstier2.sh`, Tier 2 tables in docs/qnn-vs-dff.md vs.
# the benchmark JSON snapshot) runs alongside the default suite, `nnall`,
# and `qnn`, and is also available as its own selector.
set -euo pipefail
cd "$(dirname "$0")"

run_docs_tier2_check() {
	echo "[test.sh] Verifying Tier 2 docs tables against benchmark JSON..."
	bash ./checkdocstier2.sh
}

if [[ "${1:-}" == "package-consumer" ]]; then
	exec cmake -P ../tests/package-consumer/run.cmake
fi

if [[ $# -eq 0 ]]; then
	./build/glades-unit-tests
	run_docs_tier2_check
	exit 0
fi

case "$1" in
	docs-tier2|docs|qnn-docs)
		run_docs_tier2_check
		exit 0
		;;
esac

./build/glades-unit-tests "$@"

case "$1" in
	nnall|qnn)
		run_docs_tier2_check
		;;
esac
