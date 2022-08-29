#!/bin/bash

# exit when any command fails
set -e

# Run python tests
if command -v conda &> /dev/null
then
    source activate robotai
fi

python3 -m unittest discover src.tests

# Run c++ tests
cd build
ctest