@echo off
setlocal enabledelayedexpansion

:: Usage: dev-build.bat [VS_VERSION] [cuda]
:: Examples:
::   dev-build.bat              (VS 2022, no CUDA)
::   dev-build.bat 2022 cuda    (VS 2022, with CUDA)
::
:: Dev mode copies shmea headers from ../ShmeaDB into include/
:: so they stay in sync with the source tree.

set "VS_VER=2022"
if not "%~1"=="" set "VS_VER=%~1"

set "CUDA_FLAG="
set "CMAKE_PRESET=windows-dev"
if /i "%~2"=="cuda" (
    set "CUDA_FLAG=yes"
    set "CMAKE_PRESET=windows-dev-cuda"
)

echo ============================================
echo  glades-ml - Dev Build (Windows)
echo  Visual Studio version: !VS_VER!
if defined CUDA_FLAG echo  CUDA: ENABLED
if not defined CUDA_FLAG echo  CUDA: disabled
echo  DEV_MODE: copying shmea headers from ..\ShmeaDB
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
if exist "build\CMakeCache.txt" (
    echo [FIX] Clearing stale CMake cache...
    rmdir /s /q build >nul 2>&1
)

echo [OK] VCPKG_ROOT = !VCPKG_ROOT!

:: --------------------------------------------------
:: 3. Verify ShmeaDB source and installed lib
:: --------------------------------------------------
if not exist "..\ShmeaDB\Backend" (
    echo [ERROR] ShmeaDB source not found at ..\ShmeaDB
    echo         DEV_MODE needs the ShmeaDB source tree as a sibling directory.
    exit /b 1
)
echo [OK] ShmeaDB source found at ..\ShmeaDB

if not exist "!USERPROFILE!\shmea\bin\shmea.dll" (
    echo [ERROR] shmea.dll not found at !USERPROFILE!\shmea\bin\shmea.dll
    echo         Build and install ShmeaDB first, needed for linking.
    exit /b 1
)
echo [OK] ShmeaDB installation found at !USERPROFILE!\shmea

:: --------------------------------------------------
:: 4. Verify CUDA if requested
:: --------------------------------------------------
if defined CUDA_FLAG (
    where nvcc >nul 2>&1
    if !errorlevel! neq 0 (
        echo [ERROR] CUDA requested but nvcc not found on PATH.
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
echo  Done! glades-ml dev build installed to %USERPROFILE%\glades
echo  Shmea headers copied to include\Backend\
echo ============================================

endlocal
