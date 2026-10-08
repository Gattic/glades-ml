# Windows Build Scripts for glades-ml

**Date:** 2026-03-08

## Goal

Create `.bat` build scripts for glades-ml matching ShmeaDB's pattern, with optional CUDA support.

## Scripts

### `build-and-install.bat` (root)

**Usage:** `build-and-install.bat [VS_VER] [cuda]`

- `VS_VER` — Visual Studio version directory name (default: `2022`)
- `cuda` — optional flag to enable CUDA build

**Flow:**

1. Parse args: `%1` -> VS_VER (default `2022`), `%2` -> CUDA flag
2. VS Developer Environment detection (flat `if exist` / `goto` pattern from ShmeaDB)
3. Verify `VCPKG_ROOT` + freetype fallback (same as ShmeaDB)
4. Verify shmea is installed at `%USERPROFILE%\shmea\bin\shmea.dll`
5. Verify cmake + ninja on PATH
6. Select CMake preset: `windows-release` or `windows-cuda` based on CUDA flag
7. Configure, build, install to `%USERPROFILE%\glades`

### `unit-tests/build-and-run.bat`

**Usage:** `build-and-run.bat [VS_VER] [cuda]`

**Flow:**

1. Parse args: same as above
2. VS Developer Environment detection (same)
3. Verify `VCPKG_ROOT` + freetype fallback
4. Verify shmea + glades DLLs installed
5. Set PATH for runtime DLLs (shmea, glades, freetype, CUDA libs if enabled)
6. Verify cmake + ninja on PATH
7. Select CMake preset: `windows-release` or `windows-cuda`
8. Configure, build, run `glades-unit-tests.exe`

## Additional Changes

- Add `windows-cuda` configure and build presets to `unit-tests/CMakePresets.json`
- Both scripts use `setlocal enabledelayedexpansion` and `!var!` syntax to avoid `(x86)` parsing issues

## Examples

```batch
build-and-install.bat              :: VS 2022, no CUDA
build-and-install.bat 18           :: VS 18, no CUDA
build-and-install.bat 18 cuda      :: VS 18, with CUDA
build-and-install.bat 2022 cuda    :: VS 2022, with CUDA

cd unit-tests
build-and-run.bat                  :: VS 2022, no CUDA
build-and-run.bat 18 cuda          :: VS 18, with CUDA
```
