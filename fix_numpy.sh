#!/bin/bash
# Fix numpy installation on Jetson Nano

echo "=================================="
echo "Fixing numpy on Jetson Nano"
echo "=================================="
echo ""

# Remove any pip-installed numpy (x86_64 version that doesn't work on ARM)
echo "Removing pip-installed numpy (if any)..."
python3.6 -m pip uninstall -y numpy 2>/dev/null || true

echo ""
echo "Installing ARM-compatible numpy via apt..."
sudo apt-get update
sudo apt-get install -y python3-numpy

echo ""
echo "Verifying installation..."
if python3.6 -c "import numpy; print('✓ numpy', numpy.__version__, 'works on ARM')" 2>/dev/null; then
    echo "✓ Success! numpy is now working"
else
    echo "✗ Failed. numpy still not working"
    exit 1
fi

echo ""
echo "Testing with PyTorch..."
if python3.6 -c "import torch, numpy; print('✓ Both torch and numpy work together')" 2>/dev/null; then
    echo "✓ Ready for AI training!"
else
    echo "⚠ Warning: Issue detected, but numpy should work"
fi

