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

`CMAKE_CUDA_ARCHITECTURES` defaults to `native`, which is resolved at
configure time to the compute capability of the visible GPU(s) (e.g. `8.9`
→ `89`); the resolved numbers are pinned in the cache. Every requested
architecture must be **sm_80 or newer**: the 1-bit WMMA GEMM kernel uses
`bmmaBitOpAND`, which is unavailable on sm_75 and older, so configuration
fails with an explicit error for lower targets (e.g. `75`, `70`).

Non-numeric tokens are resolved before the check, so they cannot slip past
it: `native` becomes the detected GPU architecture (configuration fails if
no GPU can be detected — set the value explicitly in that case), while
`all` and `all-major` expand to every toolkit-supported architecture and are
therefore rejected because they include sub-sm_80 targets. Use values such
as `80`, `86`, `89`, `90`, or a list like `80;86;89;90`. The `CUDAARCHS`
environment variable is honoured when the cache value is unset.

---

## Installation

make install

---

## Uninstall

make uninstall
