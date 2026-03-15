#!/bin/bash
set -e

echo "============================================"
echo " glades-ml - Build and Install (Linux)"
echo "============================================"
echo

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Verify prerequisites
command -v cmake >/dev/null 2>&1 || { echo "[ERROR] cmake not found."; exit 1; }
command -v make >/dev/null 2>&1 || { echo "[ERROR] make not found."; exit 1; }
echo "[OK] Prerequisites found."

# Verify shmea is installed
if [ ! -f "$HOME/.local/lib/libshmea.so" ]; then
    echo "[ERROR] libshmea.so not found at ~/.local/lib/libshmea.so"
    echo "        Build and install ShmeaDB first."
    exit 1
fi
echo "[OK] ShmeaDB installation found."

# Clean dev-mode include/ to ensure installed shmea headers are used
if [ -d "include/Backend" ]; then
    echo "[CLEAN] Removing dev-mode include/ ..."
    rm -rf include
fi

# Configure
mkdir -p build && cd build
echo "[STEP] Configuring..."
cmake ..
echo "[OK] Configure succeeded."
echo

# Build
echo "[STEP] Building..."
make -j"$(nproc)"
echo "[OK] Build succeeded."
echo

# Install
echo "[STEP] Installing to ~/.local ..."
make install
echo "[OK] Installed to ~/.local"
echo

echo "============================================"
echo " Done! glades-ml installed to ~/.local"
echo "============================================"
