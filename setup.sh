#!/bin/bash
# Quick start script for Flappy Bird AI Training on Jetson Nano

set -e

echo "========================================"
echo "Flappy Bird AI Training Setup"
echo "========================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Python 3.6 is installed
if ! command -v python3.6 &> /dev/null; then
    echo -e "${RED}Error: python3.6 is not installed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ python3.6 found${NC}"

# Verify pip via python3.6
if ! python3.6 -m pip --version &> /dev/null; then
    echo -e "${RED}Error: python3.6 -m pip is unavailable${NC}"
    exit 1
fi

echo -e "${GREEN}✓ python3.6 -m pip available${NC}"

# Check if cargo is installed
if ! command -v cargo &> /dev/null; then
    echo -e "${RED}Error: Cargo (Rust) is not installed${NC}"
    echo "Install from: https://rustup.rs/"
    exit 1
fi

echo -e "${GREEN}✓ Cargo found${NC}"

# Check PyTorch availability
echo ""
echo "Checking PyTorch installation..."
echo -e "${YELLOW}Note: PyTorch 1.10.0 should be pre-installed on Jetson Nano${NC}"

# Simple import check first
if python3.6 -c "import torch" 2>/dev/null; then
    echo -e "${GREEN}✓ PyTorch detected and importable${NC}"
    # Try to get version info (may fail on some systems)
    python3.6 -c "import torch; print('  Version:', torch.__version__)" 2>/dev/null || echo -e "${YELLOW}  (version check skipped)${NC}"
else
    echo -e "${RED}Error: PyTorch not found or cannot be imported${NC}"
    echo "Please install PyTorch 1.10.0 for Jetson Nano"
    echo "Download from NVIDIA's pre-built wheels for Jetson"
    exit 1
fi

# Install Python dependencies
echo ""
echo "Installing Python dependencies..."
python3.6 -m pip install -r requirements.txt --user

echo -e "${GREEN}✓ Python dependencies installed${NC}"

# Build the Rust project
echo ""
echo "Building Rust game (this may take a while)..."
cargo build --release

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Rust game built successfully${NC}"
else
    echo -e "${RED}✗ Failed to build Rust game${NC}"
    exit 1
fi

# Create necessary directories
mkdir -p models
mkdir -p plots

echo ""
echo "========================================"
echo -e "${GREEN}Setup Complete!${NC}"
echo "========================================"
echo ""
echo "Available commands:"
echo ""
echo "1. Play the game manually:"
echo "   cargo run --release"
echo ""
echo "2. Train the AI (1000 episodes):"
echo "   python3.6 train_ai.py"
echo ""
echo "3. Test a trained model:"
echo "   python3.6 test_ai.py --model models/best_model.pth"
echo ""
echo "4. Visualize training results:"
echo "   python3.6 visualize_training.py"
echo ""
echo "5. Compare all models:"
echo "   python3.6 test_ai.py --compare"
echo ""
echo "For Jetson Nano optimization, run:"
echo "   sudo nvpmodel -m 0"
echo "   sudo jetson_clocks"
echo ""
echo "========================================"
