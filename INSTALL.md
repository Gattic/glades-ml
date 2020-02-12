# Install, Compile, and Run

---

## Dependencies

Linux/Cygwin:

`cmake`

`make`

`gcc`

`g++`

`OpenSSL: libssl-dev`

Cygwin:

`xinit`

MacOS:

Install homebrew, a macOS package manager.

`/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"`
 
`brew install cmake`

Install Xcode command line tools.

`$ xcode-select --install`

`$ brew install pkgconfig`

`brew install openssl`

`brew link --force openssl` (Follow instructions to set environment paths)

`brew install sdl2`

`brew install sdl_ttf`

---

## Installation

sudo make install

---

## Compilation

Linux/MacOS/Windows/Cygwin:
```
mkdir build
cd build
cmake ..
make
```
