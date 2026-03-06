# Windows MSVC + CUDA Toolchain Migration Design

**Date:** 2026-03-06
**Scope:** ShmeaDB + glades-ml
**Status:** Approved

## Problem

- Strawberry Perl provides the current Windows toolchain (GCC/MinGW, CMake, Ninja) for ShmeaDB
- CUDA is incompatible with Strawberry Perl
- glades-ml recently added optional CUDA support and needs Windows compatibility
- Need a unified toolchain that supports both projects and enables CUDA on Windows

## Decision

Replace Strawberry Perl with MSVC (Visual Studio Build Tools) + vcpkg + Ninja. Use CMake presets for cross-platform build configuration.

### Toolchain

| Component | Old (Windows) | New (Windows) |
|-----------|--------------|---------------|
| Compiler | Strawberry Perl MinGW g++ | MSVC cl.exe (Visual Studio Build Tools) |
| Build generator | Ninja (via Strawberry) | Ninja (via VS or standalone) |
| CMake | Via Strawberry | Via VS or standalone |
| Package manager | vcpkg (partial) | vcpkg (primary) |
| CUDA host compiler | N/A | MSVC cl.exe (NVIDIA-supported) |

### Dependencies

**Windows prerequisites:**
- Visual Studio Build Tools (provides MSVC cl.exe)
- CMake (standalone or via VS)
- Ninja (standalone or via VS)
- vcpkg (for FreeType and other libs)
- CUDA Toolkit (optional, for glades-ml GPU support)

**Build order:**
1. `vcpkg install freetype`
2. ShmeaDB: `cmake --preset windows-release && ninja && ninja install`
3. glades-ml: `cmake --preset windows-cuda && ninja && ninja install`

## Phase 1: ShmeaDB — Toolchain Migration

### CMakeLists.txt changes
- Remove MinGW/GCC-specific flags under `WIN32` guards
- Add MSVC-specific flags (e.g., `/W3` instead of `-Wall -Wextra`)
- Ensure `find_package(Freetype)` works with vcpkg toolchain file
- Linux flags/paths unchanged

### CMakePresets.json (new file)
- `linux-release` — GCC, make, `~/.local` install prefix
- `windows-release` — MSVC, Ninja, vcpkg toolchain file, `%USERPROFILE%\shmea` install prefix
- Configurable vcpkg root via `VCPKG_ROOT` environment variable

### INSTALL.md update
- Rewrite Windows section: MSVC Build Tools + CMake + Ninja + vcpkg
- Remove all Strawberry Perl references
- Linux remains default/primary section
- Document vcpkg FreeType install step

### Validation
- All existing unit tests pass on Windows with MSVC

## Phase 2: glades-ml — Windows Port + CUDA

### CMakeLists.txt changes
- Add MSVC compiler flag handling (parallel to existing GCC flags)
- Update CUDA detection for Windows paths (`C:\Program Files\NVIDIA GPU Computing Toolkit\...`)
- Ensure `find_package(shmea)` resolves from `%USERPROFILE%\shmea`
- Fix any GCC-isms that MSVC rejects (VLAs, `__attribute__`, etc.)

### CUDA CMakeLists.txt changes
- glibc compatibility layer (`CudaGlibcCompat.cmake`) — skip on Windows (Linux-only concern)
- CUDA arch generation flags work cross-platform already
- Verify cuBLAS linking on Windows

### CMakePresets.json (new file)
- `linux-release` — GCC, make, no CUDA
- `linux-cuda` — GCC, make, CUDA enabled
- `windows-release` — MSVC, Ninja, vcpkg, no CUDA
- `windows-cuda` — MSVC, Ninja, vcpkg, CUDA enabled

### CLAUDE.md update
- Add dedicated Windows installation section (MSVC + vcpkg + CUDA Toolkit)
- Update build commands to reference CMake presets
- Linux remains default context for existing instructions

### Validation
- Unit tests pass on Windows with and without CUDA

## What stays the same
- All Linux build paths unchanged
- ShmeaDB platform abstractions (`platform.h`, `GMutex`, `GThread`, `GDir`) unchanged
- glades-ml ML/math code unchanged (pure C++)
- Docker builds unchanged
- `.configure.sh` scripts remain as Linux convenience wrappers
