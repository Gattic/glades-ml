# Install, Compile, and Run

Windows development installs use `%USERPROFILE%\dev\installed` via
`dev-build.bat` / `windows-dev`. Release installs use `C:\GatticSDK` via
`build-and-install.bat` / `windows-release`. Build ShmeaDB into the same prefix
first. See [the shared build guide](../InformationGattic/build-guide.md).

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

## Linux build and tests

Use CMake 3.25 or newer for the presets, Make, a C++23 compiler, and FreeType
headers. Build and install sibling ShmeaDB first. From this repository's root:

```sh
bash build-and-install.sh                 # CPU build and install
bash dev-build.sh                         # CPU build with source Shmea headers
bash unit-tests/build-and-run.sh cv        # build tests and run a focused selector
bash unit-tests/dev-run.sh gan             # after dev-build.sh
bash unit-tests/test.sh numerical-edge     # reuse the built test executable
bash unit-tests/test.sh package-consumer   # build/install consumer and model reads
```

The test build helper defaults to `nnall`. CPU-only builds do not run GPU suites.
With an installed CUDA toolkit and supported host compiler, append `cuda` to
`build-and-install.sh`, `dev-build.sh`, `unit-tests/build-and-run.sh`, or
`unit-tests/dev-run.sh`. CUDA device compilation uses C++17. Switching back to
the CPU preset explicitly disables CUDA.

Installations default to `$HOME/.local`. The unit-test helpers use this repository's
`build/gladesConfig.cmake`, so tests exercise the library just built. The root
convenience header is installed as `glades/main.h`, leaving ShmeaDB's `main.h`
intact. The package also provides the namespaced ML header tree.

All helpers accept quoted CMake `-DNAME=VALUE` overrides and stop on failure.
For a custom dependency/install prefix, for example:

```sh
bash build-and-install.sh -DCMAKE_INSTALL_PREFIX="$HOME/libs" -DCMAKE_PREFIX_PATH="$HOME/libs"
```

When changing from an existing cached dependency, also pass
`-Dshmea_DIR="$HOME/libs/share/shmea/cmake"`. Use
`CMAKE_BUILD_PARALLEL_LEVEL=4` to adjust build concurrency (default: 8).
Run helpers from any working directory; they resolve their own repository path.
Builds reuse `build/` incrementally. Development headers for glades-ml and
gfxplusplus live under `build/shmea-include/` and do not modify source `include/`
folders. ShmeaDB still needs to be built and installed for linking.

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

Copies shmea headers from the ShmeaDB source tree (expected at `..\ShmeaDB`) into `build/shmea-include/`. Shmea must still be installed for linking.

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
| `build/shmea-include/` folder | Not created | Created with fresh headers |
| Use case | CI, releases, end users | Active development |

The `build/shmea-include/` directory is gitignored and updated on each dev configure.
