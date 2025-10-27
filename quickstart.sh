#!/usr/bin/env bash
# Quick start script for BiDoRA

set -e

echo "=========================================="
echo "BiDoRA Quick Start"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Install package
echo ""
echo "Installing bidora package..."
if command -v uv &> /dev/null; then
    echo "Using uv for installation..."
    uv pip install -e .
else
    echo "Using pip for installation..."
    pip install -e .
fi

# Show hardware info
echo ""
echo "=========================================="
echo "Hardware Information"
echo "=========================================="
bidora info

# List recommended models
echo ""
echo "=========================================="
echo "Recommended Models"
echo "=========================================="
bidora list-models

# Check if example data exists
if [ ! -f "examples/rust_3d_train.jsonl" ]; then
    echo ""
    echo "⚠️  Example training data not found at examples/rust_3d_train.jsonl"
    echo "Please create training data before running training."
    exit 0
fi

# Offer to run training
echo ""
echo "=========================================="
echo "Ready to train!"
echo "=========================================="
echo ""
echo "Example command:"
echo ""
echo "  bidora train \\"
echo "    --train-file examples/rust_3d_train.jsonl \\"
echo "    --model Qwen/Qwen2.5-Coder-7B-Instruct \\"
echo "    --output ./output \\"
echo "    --rank 8 \\"
echo "    --epochs 3"
echo ""
read -p "Run example training now? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Starting training..."
    bidora train \
        --train-file examples/rust_3d_train.jsonl \
        --model Qwen/Qwen2.5-Coder-7B-Instruct \
        --output ./output \
        --rank 8 \
        --epochs 3 \
        --max-samples 50
fi

echo ""
echo "Quick start complete!"
