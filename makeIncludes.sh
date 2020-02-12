#!/bin/bash
# makeIncludes.sh - quick include builder for libraries
# -----------------------------------------------------

if [ -d "include/" ]; then
  rm -rf include/*
else
  mkdir include/
fi

find . -type f | grep -i "\.h$" | xargs -i cp --parents "{}" include
