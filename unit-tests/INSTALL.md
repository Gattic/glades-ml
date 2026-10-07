# Unit Tests - Install, Compile, and Run

---

## Prerequisites

glades-ml and shmea must be built and installed before running unit tests.

---

## Linux

Build and install ShmeaDB and build the main glades-ml library first. From the
repository root:

```sh
bash build-and-install.sh
bash unit-tests/build-and-run.sh cv
```

For development headers:

```sh
bash dev-build.sh
bash unit-tests/dev-run.sh gan
```

The helpers accept `cuda`, selectors, and quoted CMake `-DNAME=VALUE` overrides.
The default selector is `nnall`; GPU suites require a CUDA build. Development
headers live in the main `build/shmea-include/`. For additional focused tests,
reuse the executable via `bash unit-tests/test.sh <selector>`.
See [the library installation guide](../INSTALL.md) for prerequisites and custom
prefixes.

---

## Windows

### Prod

Uses installed shmea and glades headers. If a dev-mode `include/` exists, it is automatically removed to ensure a clean prod build.

```powershell
.\build-and-run.bat
```

With CUDA:
```powershell
.\build-and-run.bat 2022 cuda
```

### Dev vs Prod

| | Prod (`build-and-run.bat`) | Dev (`cmake .. -DDEV_MODE=ON`) |
|---|---|---|
| Shmea headers | From installed shmea | From ShmeaDB source tree via `build/shmea-include/` |
| Cleans `include/` | Yes, automatically | No |
| Use case | CI, releases | Active development |
