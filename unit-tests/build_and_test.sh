#!/bin/bash

# Clean the build directory
rm -rf ./build/

# Navigate to the build directory
mkdir build/
cd ./build/

# Configure the CMake build
cmake ..

# Build the project
make run
