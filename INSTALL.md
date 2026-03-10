# Install, Compile, and Run

---

## Dependencies

### Debian

`cmake`

`make`

`g++`

`libfreetype6-dev`

sudo dnf install cuda

### Fedora

sudo dnf install -y gcc gcc-c++ clang cmake make
sudo dnf install -y freetype-devel
sudo apt install nvidia-cuda-toolkit
sudo dnf install -y libasan

### Windows

See the [Windows](#windows-build) section below.

---

## Linux Build

### Compilation (prod)

Requires shmea to be installed (`make install` from ShmeaDB).

```
mkdir build
cd build
cmake ..
make
```

Or using presets:
```
cmake --preset linux-release
cmake --build --preset linux-release
```

### Compilation (dev)

Dev mode copies shmea headers from the ShmeaDB source tree into `include/` so they stay in sync. Shmea must still be installed for linking.

```
mkdir build
cd build
cmake .. -DDEV_MODE=ON
make
```

By default, `SHMEA_SOURCE_DIR` points to `../ShmeaDB`. Override it if your ShmeaDB source is elsewhere:
```
cmake .. -DDEV_MODE=ON -DSHMEA_SOURCE_DIR=/path/to/ShmeaDB
```

### Installation

```
make install
```

### Uninstall

```
make uninstall
```

---

## Windows Build

### Prerequisites

1. **Visual Studio 2022 Build Tools** with the "Desktop development with C++" workload
2. **vcpkg** with `VCPKG_ROOT` environment variable set:
   ```powershell
   vcpkg install freetype:x64-windows
   ```
3. **Ninja** (included with VS Build Tools or install separately)
4. **ShmeaDB** built and installed:
   ```powershell
   cd ShmeaDB
   .\build-and-install.bat
   ```

### Prod build

Uses shmea headers from the installed ShmeaDB.

```powershell
.\build-and-install.bat
```

With CUDA:
```powershell
.\build-and-install.bat 2022 cuda
```

### Dev build

Copies shmea headers from the ShmeaDB source tree (expected at `..\ShmeaDB`) into `include/`. Shmea must still be installed for linking.

```powershell
.\dev-build.bat
```

With CUDA:
```powershell
.\dev-build.bat 2022 cuda
```

### Dev vs Prod

| | Prod (`build-and-install.bat`) | Dev (`dev-build.bat`) |
|---|---|---|
| Shmea headers | From installed shmea | Copied from ShmeaDB source tree |
| Shmea library | Installed (`shmea.dll`) | Installed (`shmea.dll`) |
| `include/` folder | Not created | Created with fresh headers |
| Use case | CI, releases, end users | Active development |

The `include/` directory is gitignored and regenerated on each dev configure.
