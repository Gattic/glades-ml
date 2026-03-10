# Unit Tests - Install, Compile, and Run

---

## Prerequisites

glades-ml and shmea must be built and installed before running unit tests.

---

## Linux

### Prod (uses installed shmea headers)

```
mkdir build
cd build
cmake ..
make run
```

### Dev (uses shmea headers from source tree)

Run the main project's dev build first to populate `include/`:
```
cd ..
mkdir build && cd build
cmake .. -DDEV_MODE=ON
make
cd ../unit-tests
```

Then build and run the unit tests with dev mode:
```
mkdir build
cd build
cmake .. -DDEV_MODE=ON
make run
```

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
| Shmea headers | From installed shmea | From ShmeaDB source tree via `include/` |
| Cleans `include/` | Yes, automatically | No |
| Use case | CI, releases | Active development |
