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

---

## Compilation

```
sh .configure.sh
```
or
```
mkdir build
cd build
cmake ..
make
```

### CUDA builds

The CUDA backend is optional and off by default:

```
sh .configure.sh cuda
# or, manually:
mkdir build && cd build
cmake .. -DGLADES_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89
make -j$(nproc)
```

`CMAKE_CUDA_ARCHITECTURES` defaults to `native` when unset. Every requested
architecture must be **sm_80 or newer**: the 1-bit WMMA GEMM kernel uses
`bmmaBitOpAND`, which is unavailable on sm_75 and older, so configuration
fails with an explicit error for lower targets (e.g. `75`, `70`). Use values
such as `80`, `86`, `89`, `90`, or a list like `80;86;89;90`.

---

## Installation

make install

---

## Uninstall

make uninstall
