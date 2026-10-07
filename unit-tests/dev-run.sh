#!/usr/bin/env bash
# Linux counterpart to dev-run.bat; test saves stay in unit-tests/.
set -euo pipefail
source_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$source_dir"
preset="linux-dev"
cmake_args=()
test_args=()
for arg in "$@"; do
    case "$arg" in
        cuda|--cuda) preset="linux-dev-cuda" ;;
        -D?*=*) cmake_args+=("$arg") ;;
        --help|-h) echo "Usage: bash dev-run.sh [cuda] [test-selector] [-DNAME=VALUE ...]"; exit 0 ;;
        -*) echo "Unknown option: $arg" >&2; exit 2 ;;
        *) test_args+=("$arg") ;;
    esac
done
cmake_args+=("-Dglades_DIR=$source_dir/../build")
cmake --preset "$preset" "${cmake_args[@]}"
cmake --build --preset "$preset" --parallel "${CMAKE_BUILD_PARALLEL_LEVEL:-8}"
if [ "${#test_args[@]}" -eq 0 ]; then test_args=(nnall); fi
bash test.sh "${test_args[@]}"
