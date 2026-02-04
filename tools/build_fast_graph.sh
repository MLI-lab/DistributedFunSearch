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

# Create symlinks for multiarch glibc headers (Ubuntu/Debian)
# This fixes "bits/libc-header-start.h: No such file or directory" errors
if [ -d "/usr/include/x86_64-linux-gnu/bits" ] && [ ! -e "/usr/include/bits" ]; then
    echo "Creating symlinks for multiarch headers..."
    ln -sf /usr/include/x86_64-linux-gnu/bits /usr/include/bits 2>/dev/null || \
        sudo ln -sf /usr/include/x86_64-linux-gnu/bits /usr/include/bits
    ln -sf /usr/include/x86_64-linux-gnu/gnu /usr/include/gnu 2>/dev/null || \
        sudo ln -sf /usr/include/x86_64-linux-gnu/gnu /usr/include/gnu
    ln -sf /usr/include/x86_64-linux-gnu/sys /usr/include/sys 2>/dev/null || \
        sudo ln -sf /usr/include/x86_64-linux-gnu/sys /usr/include/sys
fi

cd "$CPP_SRC"

# Clean old build artifacts and .so files
rm -rf "$CPP_SRC"/build "$CPP_SRC"/fast_graph_cpp*.so "$UTILS_DIR"/fast_graph_cpp*.so

# Use system compiler to avoid conda compiler / glibc header conflicts
CC=/usr/bin/gcc CXX=/usr/bin/g++ python3 setup.py build_ext --inplace

# Copy to utils directory
cp fast_graph_cpp*.so "$UTILS_DIR/"
echo "Installed: $(ls "$UTILS_DIR"/fast_graph_cpp*.so)"

# Verify
python3 -c "from disfun.utils.fast_graph import USING_CPP; print(f'USING_CPP: {USING_CPP}')"
