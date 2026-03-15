#!/bin/bash
set -e

echo "============================================"
echo " glades-ml - Dev Build (Linux)"
echo " DEV_MODE: copying shmea headers from ../ShmeaDB"
echo "============================================"
echo

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Verify prerequisites
command -v cmake >/dev/null 2>&1 || { echo "[ERROR] cmake not found."; exit 1; }
command -v make >/dev/null 2>&1 || { echo "[ERROR] make not found."; exit 1; }
echo "[OK] Prerequisites found."

# Verify ShmeaDB source tree
if [ ! -d "../ShmeaDB/Backend" ]; then
    echo "[ERROR] ShmeaDB source not found at ../ShmeaDB"
    echo "        DEV_MODE needs the ShmeaDB source tree as a sibling directory."
    exit 1
fi
echo "[OK] ShmeaDB source found at ../ShmeaDB"

# Verify shmea is installed (still needed for linking)
if [ ! -f "$HOME/.local/lib/libshmea.so" ]; then
    echo "[ERROR] libshmea.so not found at ~/.local/lib/libshmea.so"
    echo "        Build and install ShmeaDB first (needed for linking)."
    exit 1
fi
echo "[OK] ShmeaDB installation found."

# Configure with DEV_MODE
mkdir -p build && cd build
echo "[STEP] Configuring with DEV_MODE=ON..."
cmake .. -DDEV_MODE=ON
echo "[OK] Configure succeeded."
echo

# Build
echo "[STEP] Building..."
make -j"$(nproc)"
echo "[OK] Build succeeded."
echo

echo "============================================"
echo " Done! glades-ml dev build complete."
echo " Shmea headers copied to include/Backend/"
echo "============================================"
