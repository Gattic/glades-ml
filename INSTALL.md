# Install, Compile, and Run

---

## Dependencies

### Debian

`cmake`

`make`

`g++`

`libfreetype6-dev`

### Fedora

sudo dnf install -y gcc gcc-c++ clang cmake make
sudo dnf install -y freetype-devel
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
