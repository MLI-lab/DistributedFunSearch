#!/bin/bash
# Build FastGraph C++ extension for the current Python version.
# Run this after pip install if using graph-based problems.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CPP_SRC="$PROJECT_ROOT/src/disfun/utils/_fast_graph_cpp_src"
UTILS_DIR="$PROJECT_ROOT/src/disfun/utils"

echo "Building FastGraph C++ extension..."
echo "Python: $(python3 --version)"

cd "$CPP_SRC"
python3 setup.py build_ext --inplace

# Copy .so to utils directory
cp fast_graph_cpp*.so "$UTILS_DIR/"
echo "Installed: $(ls "$UTILS_DIR"/fast_graph_cpp*.so)"

# Verify
python3 -c "from disfun.utils.fast_graph import USING_CPP; print(f'USING_CPP: {USING_CPP}')"
