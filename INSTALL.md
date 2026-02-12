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

---

## Installation

make install

---

## Uninstall

make uninstall
