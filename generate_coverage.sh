#!/bin/bash

# Find all .gcda files and run gcov on each with --preserve-paths
find ./cmake-build-relwithdebinfo -name "*.gcda" -exec gcov --preserve-paths {} \;

echo "Coverage data generated."
