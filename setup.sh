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

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is not installed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Python 3 found${NC}"

# Check if pip3 is installed
if ! command -v pip3 &> /dev/null; then
    echo -e "${RED}Error: pip3 is not installed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ pip3 found${NC}"

# Check if cargo is installed
if ! command -v cargo &> /dev/null; then
    echo -e "${RED}Error: Cargo (Rust) is not installed${NC}"
    echo "Install from: https://rustup.rs/"
    exit 1
fi

echo -e "${GREEN}✓ Cargo found${NC}"

# Check CUDA availability
echo ""
echo "Checking CUDA availability..."
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')" 2>/dev/null || {
    echo -e "${YELLOW}Warning: Could not check CUDA. Installing dependencies...${NC}"
}

# Install Python dependencies
echo ""
echo "Installing Python dependencies..."
pip3 install -r requirements.txt --user

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
echo "   python3 train_ai.py"
echo ""
echo "3. Test a trained model:"
echo "   python3 test_ai.py --model models/best_model.pth"
echo ""
echo "4. Visualize training results:"
echo "   python3 visualize_training.py"
echo ""
echo "5. Compare all models:"
echo "   python3 test_ai.py --compare"
echo ""
echo "For Jetson Nano optimization, run:"
echo "   sudo nvpmodel -m 0"
echo "   sudo jetson_clocks"
echo ""
echo "========================================"

