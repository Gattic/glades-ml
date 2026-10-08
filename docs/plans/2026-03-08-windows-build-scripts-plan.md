# Windows Build Scripts Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create `.bat` build scripts for glades-ml with optional CUDA support, mirroring ShmeaDB's scripts.

**Architecture:** Two `.bat` files (`build-and-install.bat` at root, `build-and-run.bat` in `unit-tests/`) accepting `[VS_VER] [cuda]` arguments. Both use `setlocal enabledelayedexpansion` and `!var!` syntax to avoid `(x86)` parsing issues. A `windows-cuda` preset is added to `unit-tests/CMakePresets.json`.

**Tech Stack:** Windows batch scripting, CMake presets, MSVC toolchain, vcpkg, CUDA (optional)

---

### Task 1: Add windows-cuda preset to unit-tests/CMakePresets.json

**Files:**
- Modify: `unit-tests/CMakePresets.json`

**Step 1: Add the windows-cuda configure and build presets**

Add a `windows-cuda` configure preset that inherits `windows-release` and sets `GLADES_ENABLE_CUDA=ON`, plus a matching build preset. This mirrors the pattern already in the root `CMakePresets.json`.

```json
{
    "version": 6,
    "configurePresets": [
        {
            "name": "linux-release",
            "displayName": "Linux Release",
            "generator": "Unix Makefiles",
            "binaryDir": "${sourceDir}/build",
            "cacheVariables": {
                "CMAKE_BUILD_TYPE": "Release"
            },
            "condition": {
                "type": "notEquals",
                "lhs": "${hostSystemName}",
                "rhs": "Windows"
            }
        },
        {
            "name": "windows-release",
            "displayName": "Windows Release (MSVC)",
            "generator": "Ninja",
            "binaryDir": "${sourceDir}/build",
            "cacheVariables": {
                "CMAKE_BUILD_TYPE": "Release",
                "CMAKE_C_COMPILER": "cl.exe",
                "CMAKE_CXX_COMPILER": "cl.exe",
                "CMAKE_TOOLCHAIN_FILE": "$env{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake",
                "CMAKE_MSVC_RUNTIME_LIBRARY": "MultiThreadedDLL"
            },
            "condition": {
                "type": "equals",
                "lhs": "${hostSystemName}",
                "rhs": "Windows"
            }
        },
        {
            "name": "windows-cuda",
            "displayName": "Windows Release + CUDA (MSVC)",
            "inherits": "windows-release",
            "cacheVariables": {
                "GLADES_ENABLE_CUDA": "ON"
            }
        }
    ],
    "buildPresets": [
        {
            "name": "linux-release",
            "configurePreset": "linux-release"
        },
        {
            "name": "windows-release",
            "configurePreset": "windows-release"
        },
        {
            "name": "windows-cuda",
            "configurePreset": "windows-cuda"
        }
    ]
}
```

**Step 2: Verify JSON is valid**

Run: `python -c "import json; json.load(open('unit-tests/CMakePresets.json'))"`
Expected: No output (valid JSON)

---

### Task 2: Create glades-ml build-and-install.bat

**Files:**
- Create: `build-and-install.bat`

**Step 1: Write build-and-install.bat**

Based on ShmeaDB's `build-and-install.bat` with these differences:
- Accepts second arg `cuda` to select `windows-cuda` preset
- Checks shmea is installed before building
- Installs to `%USERPROFILE%\glades`
- Displays CUDA status in banner

```batch
@echo off
setlocal enabledelayedexpansion

:: Usage: build-and-install.bat [VS_VERSION] [cuda]
:: Examples:
::   build-and-install.bat              (VS 2022, no CUDA)
::   build-and-install.bat 18           (VS 18, no CUDA)
::   build-and-install.bat 18 cuda      (VS 18, with CUDA)
::   build-and-install.bat 2022 cuda    (VS 2022, with CUDA)

set "VS_VER=2022"
if not "%~1"=="" set "VS_VER=%~1"

set "CUDA_FLAG="
set "CMAKE_PRESET=windows-release"
if /i "%~2"=="cuda" (
    set "CUDA_FLAG=yes"
    set "CMAKE_PRESET=windows-cuda"
)

echo ============================================
echo  glades-ml - Build and Install (Windows)
echo  Visual Studio version: !VS_VER!
if defined CUDA_FLAG echo  CUDA: ENABLED
if not defined CUDA_FLAG echo  CUDA: disabled
echo ============================================
echo.

:: --------------------------------------------------
:: 1. Initialize VS Developer Environment if needed
:: --------------------------------------------------
where cl.exe >nul 2>&1
if !errorlevel! equ 0 (
    echo [OK] cl.exe already available.
    goto :vcpkg_check
)
echo [INFO] cl.exe not found. Initializing VS Developer Environment...
if exist "C:\Program Files (x86)\Microsoft Visual Studio\!VS_VER!\BuildTools\Common7\Tools\VsDevCmd.bat" (
    call "C:\Program Files (x86)\Microsoft Visual Studio\!VS_VER!\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=amd64 >nul 2>&1
    echo [OK] VS Developer Environment initialized.
    goto :vcpkg_check
)
if exist "C:\Program Files\Microsoft Visual Studio\!VS_VER!\Community\Common7\Tools\VsDevCmd.bat" (
    call "C:\Program Files\Microsoft Visual Studio\!VS_VER!\Community\Common7\Tools\VsDevCmd.bat" -arch=amd64 >nul 2>&1
    echo [OK] VS Developer Environment initialized.
    goto :vcpkg_check
)
echo [ERROR] Could not find Visual Studio !VS_VER! Build Tools or Community edition.
echo         Install VS Build Tools with "Desktop development with C++" workload.
echo         See INSTALL.md for details.
exit /b 1

:vcpkg_check

:: --------------------------------------------------
:: 2. Verify VCPKG_ROOT and fix if overridden by VS
:: --------------------------------------------------
if not defined VCPKG_ROOT (
    echo [ERROR] VCPKG_ROOT is not set.
    echo         Set it to your vcpkg installation, e.g.:
    echo           set VCPKG_ROOT=C:\vcpkg
    exit /b 1
)

if exist "!VCPKG_ROOT!\installed\x64-windows\lib\freetype.lib" goto :freetype_ok
echo [WARN] Freetype not found at !VCPKG_ROOT!\installed\x64-windows\lib\freetype.lib
echo        The VS Developer Shell may have overridden VCPKG_ROOT.
echo.
if exist "C:\vcpkg\installed\x64-windows\lib\freetype.lib" (
    echo [FIX] Found freetype at C:\vcpkg. Resetting VCPKG_ROOT.
    set "VCPKG_ROOT=C:\vcpkg"
    goto :freetype_ok
)
if exist "%USERPROFILE%\vcpkg\installed\x64-windows\lib\freetype.lib" (
    echo [FIX] Found freetype at %USERPROFILE%\vcpkg. Resetting VCPKG_ROOT.
    set "VCPKG_ROOT=%USERPROFILE%\vcpkg"
    goto :freetype_ok
)
if exist "%USERPROFILE%\dev\vcpkg\installed\x64-windows\lib\freetype.lib" (
    echo [FIX] Found freetype at %USERPROFILE%\dev\vcpkg. Resetting VCPKG_ROOT.
    set "VCPKG_ROOT=%USERPROFILE%\dev\vcpkg"
    goto :freetype_ok
)
echo [ERROR] Could not find freetype in any known vcpkg location.
echo         Install it with: vcpkg install freetype:x64-windows
exit /b 1

:freetype_ok
:: If VCPKG_ROOT was reset, clear any stale CMake cache that points to the old path
if exist "build\CMakeCache.txt" (
    echo [FIX] Clearing stale CMake cache...
    rmdir /s /q build >nul 2>&1
)

echo [OK] VCPKG_ROOT = !VCPKG_ROOT!

:: --------------------------------------------------
:: 3. Verify ShmeaDB is installed
:: --------------------------------------------------
if not exist "%USERPROFILE%\shmea\bin\shmea.dll" (
    echo [ERROR] shmea.dll not found at %USERPROFILE%\shmea\bin\shmea.dll
    echo         Build and install ShmeaDB first.
    exit /b 1
)
echo [OK] ShmeaDB installation found at %USERPROFILE%\shmea

:: --------------------------------------------------
:: 4. Verify CUDA if requested
:: --------------------------------------------------
if defined CUDA_FLAG (
    where nvcc >nul 2>&1
    if !errorlevel! neq 0 (
        echo [ERROR] CUDA requested but nvcc not found on PATH.
        echo         Install CUDA Toolkit and ensure it is on PATH.
        exit /b 1
    )
    echo [OK] CUDA compiler found.
)

:: --------------------------------------------------
:: 5. Verify prerequisites
:: --------------------------------------------------
where cmake >nul 2>&1
if !errorlevel! neq 0 (
    echo [ERROR] cmake not found on PATH.
    exit /b 1
)
echo [OK] cmake found.

where ninja >nul 2>&1
if !errorlevel! neq 0 (
    echo [ERROR] ninja not found on PATH.
    exit /b 1
)
echo [OK] ninja found.

echo.

:: --------------------------------------------------
:: 6. Configure
:: --------------------------------------------------
echo [STEP] Configuring with CMake preset '!CMAKE_PRESET!'...
cmake --preset !CMAKE_PRESET!
if !errorlevel! neq 0 (
    echo [ERROR] CMake configure failed.
    exit /b 1
)
echo [OK] Configure succeeded.
echo.

:: --------------------------------------------------
:: 7. Build
:: --------------------------------------------------
echo [STEP] Building...
cmake --build --preset !CMAKE_PRESET!
if !errorlevel! neq 0 (
    echo [ERROR] Build failed.
    exit /b 1
)
echo [OK] Build succeeded.
echo.

:: --------------------------------------------------
:: 8. Install
:: --------------------------------------------------
echo [STEP] Installing to %USERPROFILE%\glades ...
cmake --install build
if !errorlevel! neq 0 (
    echo [ERROR] Install failed.
    exit /b 1
)
echo [OK] Installed to %USERPROFILE%\glades
echo.

echo ============================================
echo  Done! glades-ml installed to %USERPROFILE%\glades
echo ============================================

endlocal
```

---

### Task 3: Create glades-ml unit-tests/build-and-run.bat

**Files:**
- Create: `unit-tests/build-and-run.bat`

**Step 1: Write unit-tests/build-and-run.bat**

Based on ShmeaDB's `unit-tests/build-and-run.bat` with these differences:
- Accepts second arg `cuda` to select `windows-cuda` preset
- Checks both shmea AND glades DLLs are installed
- Adds glades DLL path to runtime PATH
- Adds CUDA libs to PATH if CUDA enabled
- Runs `glades-unit-tests.exe` instead of `shmea-unit-tests.exe`

```batch
@echo off
setlocal enabledelayedexpansion

:: Usage: build-and-run.bat [VS_VERSION] [cuda]
:: Examples:
::   build-and-run.bat              (VS 2022, no CUDA)
::   build-and-run.bat 18           (VS 18, no CUDA)
::   build-and-run.bat 18 cuda      (VS 18, with CUDA)
::   build-and-run.bat 2022 cuda    (VS 2022, with CUDA)

set "VS_VER=2022"
if not "%~1"=="" set "VS_VER=%~1"

set "CUDA_FLAG="
set "CMAKE_PRESET=windows-release"
if /i "%~2"=="cuda" (
    set "CUDA_FLAG=yes"
    set "CMAKE_PRESET=windows-cuda"
)

echo ============================================
echo  glades-ml Unit Tests - Build and Run (Windows)
echo  Visual Studio version: !VS_VER!
if defined CUDA_FLAG echo  CUDA: ENABLED
if not defined CUDA_FLAG echo  CUDA: disabled
echo ============================================
echo.

:: --------------------------------------------------
:: 1. Initialize VS Developer Environment if needed
:: --------------------------------------------------
where cl.exe >nul 2>&1
if !errorlevel! equ 0 (
    echo [OK] cl.exe already available.
    goto :vcpkg_check
)
echo [INFO] cl.exe not found. Initializing VS Developer Environment...
if exist "C:\Program Files (x86)\Microsoft Visual Studio\!VS_VER!\BuildTools\Common7\Tools\VsDevCmd.bat" (
    call "C:\Program Files (x86)\Microsoft Visual Studio\!VS_VER!\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=amd64 >nul 2>&1
    echo [OK] VS Developer Environment initialized.
    goto :vcpkg_check
)
if exist "C:\Program Files\Microsoft Visual Studio\!VS_VER!\Community\Common7\Tools\VsDevCmd.bat" (
    call "C:\Program Files\Microsoft Visual Studio\!VS_VER!\Community\Common7\Tools\VsDevCmd.bat" -arch=amd64 >nul 2>&1
    echo [OK] VS Developer Environment initialized.
    goto :vcpkg_check
)
echo [ERROR] Could not find Visual Studio !VS_VER! Build Tools or Community edition.
echo         Install VS Build Tools with "Desktop development with C++" workload.
echo         See INSTALL.md for details.
exit /b 1

:vcpkg_check

:: --------------------------------------------------
:: 2. Verify VCPKG_ROOT and fix if overridden by VS
:: --------------------------------------------------
if not defined VCPKG_ROOT (
    echo [ERROR] VCPKG_ROOT is not set.
    echo         Set it to your vcpkg installation, e.g.:
    echo           set VCPKG_ROOT=C:\vcpkg
    exit /b 1
)

if exist "!VCPKG_ROOT!\installed\x64-windows\lib\freetype.lib" goto :freetype_ok
echo [WARN] Freetype not found at !VCPKG_ROOT!\installed\x64-windows\lib\freetype.lib
echo        The VS Developer Shell may have overridden VCPKG_ROOT.
echo.
if exist "C:\vcpkg\installed\x64-windows\lib\freetype.lib" (
    echo [FIX] Found freetype at C:\vcpkg. Resetting VCPKG_ROOT.
    set "VCPKG_ROOT=C:\vcpkg"
    goto :freetype_ok
)
if exist "%USERPROFILE%\vcpkg\installed\x64-windows\lib\freetype.lib" (
    echo [FIX] Found freetype at %USERPROFILE%\vcpkg. Resetting VCPKG_ROOT.
    set "VCPKG_ROOT=%USERPROFILE%\vcpkg"
    goto :freetype_ok
)
if exist "%USERPROFILE%\dev\vcpkg\installed\x64-windows\lib\freetype.lib" (
    echo [FIX] Found freetype at %USERPROFILE%\dev\vcpkg. Resetting VCPKG_ROOT.
    set "VCPKG_ROOT=%USERPROFILE%\dev\vcpkg"
    goto :freetype_ok
)
echo [ERROR] Could not find freetype in any known vcpkg location.
echo         Install it with: vcpkg install freetype:x64-windows
exit /b 1

:freetype_ok

echo [OK] VCPKG_ROOT = !VCPKG_ROOT!

:: --------------------------------------------------
:: 3. Verify ShmeaDB and glades-ml are installed
:: --------------------------------------------------
if not exist "%USERPROFILE%\shmea\bin\shmea.dll" (
    echo [ERROR] shmea.dll not found at %USERPROFILE%\shmea\bin\shmea.dll
    echo         Build and install ShmeaDB first.
    exit /b 1
)
echo [OK] ShmeaDB installation found at %USERPROFILE%\shmea

if not exist "%USERPROFILE%\glades\bin\glades.dll" (
    echo [ERROR] glades.dll not found at %USERPROFILE%\glades\bin\glades.dll
    echo         Build and install glades-ml first -- run build-and-install.bat from the root.
    exit /b 1
)
echo [OK] glades-ml installation found at %USERPROFILE%\glades

:: --------------------------------------------------
:: 4. Verify CUDA if requested
:: --------------------------------------------------
if defined CUDA_FLAG (
    where nvcc >nul 2>&1
    if !errorlevel! neq 0 (
        echo [ERROR] CUDA requested but nvcc not found on PATH.
        echo         Install CUDA Toolkit and ensure it is on PATH.
        exit /b 1
    )
    echo [OK] CUDA compiler found.
)

:: --------------------------------------------------
:: 5. Set PATH so DLLs are found at runtime
:: --------------------------------------------------
set "PATH=%USERPROFILE%\shmea\bin;%USERPROFILE%\glades\bin;!VCPKG_ROOT!\installed\x64-windows\bin;!PATH!"
echo [OK] PATH updated with shmea.dll, glades.dll, and freetype.dll locations.
if defined CUDA_FLAG (
    if defined CUDA_PATH (
        set "PATH=!CUDA_PATH!\bin;!PATH!"
        echo [OK] PATH updated with CUDA libraries.
    )
)

:: --------------------------------------------------
:: 6. Verify prerequisites
:: --------------------------------------------------
where cmake >nul 2>&1
if !errorlevel! neq 0 (
    echo [ERROR] cmake not found on PATH.
    exit /b 1
)
echo [OK] cmake found.

where ninja >nul 2>&1
if !errorlevel! neq 0 (
    echo [ERROR] ninja not found on PATH.
    exit /b 1
)
echo [OK] ninja found.

echo.

:: --------------------------------------------------
:: 7. Configure
:: --------------------------------------------------
echo [STEP] Configuring unit tests with CMake preset '!CMAKE_PRESET!'...
cmake --preset !CMAKE_PRESET!
if !errorlevel! neq 0 (
    echo [ERROR] CMake configure failed.
    exit /b 1
)
echo [OK] Configure succeeded.
echo.

:: --------------------------------------------------
:: 8. Build
:: --------------------------------------------------
echo [STEP] Building unit tests...
cmake --build --preset !CMAKE_PRESET!
if !errorlevel! neq 0 (
    echo [ERROR] Build failed.
    exit /b 1
)
echo [OK] Build succeeded.
echo.

:: --------------------------------------------------
:: 9. Run
:: --------------------------------------------------
echo [STEP] Running unit tests...
echo.
.\build\glades-unit-tests.exe
if !errorlevel! neq 0 (
    echo.
    echo [FAIL] Unit tests failed with exit code !errorlevel!.
    exit /b !errorlevel!
)
echo.
echo ============================================
echo  All unit tests passed!
echo ============================================

endlocal
```

---

### Task 4: Verify scripts run (no CUDA)

**Step 1: Run build-and-install.bat on the default preset**

Run: `cd C:\Users\Matt\dev\glades-ml && .\build-and-install.bat`
Expected: Configures, builds, and installs to `%USERPROFILE%\glades`

**Step 2: Run build-and-run.bat on the default preset**

Run: `cd C:\Users\Matt\dev\glades-ml\unit-tests && .\build-and-run.bat`
Expected: Configures, builds, and runs `glades-unit-tests.exe`
