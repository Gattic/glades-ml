#!/usr/bin/env bash
# Linux counterpart to dev-build.bat. Extra -D options go to CMake.
set -euo pipefail
source_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$source_dir"
command -v cmake >/dev/null || { echo "CMake is required" >&2; exit 1; }
preset="linux-dev"
cmake_args=()
for arg in "$@"; do
    case "$arg" in
        cuda|--cuda) preset="linux-dev-cuda" ;;
        -D?*=*) cmake_args+=("$arg") ;;
        --help|-h) echo "Usage: bash dev-build.sh [cuda] [-DNAME=VALUE ...]"; exit 0 ;;
        *) echo "Unknown argument: $arg" >&2; exit 2 ;;
    esac
done
cmake --preset "$preset" "${cmake_args[@]}"
cmake --build --preset "$preset" --parallel "${CMAKE_BUILD_PARALLEL_LEVEL:-8}"
cmake --install build
