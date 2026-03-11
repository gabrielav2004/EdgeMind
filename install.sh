#!/bin/bash
set -e

echo "========================================"
echo "EdgeMind Installer"
echo "========================================"

# check python
if ! command -v python3 &> /dev/null; then
    echo "✗ python3 not found. please install python 3.9+"
    exit 1
fi

echo "✓ python3 found: $(python3 --version)"

# install CPU-only torch first to avoid CUDA bloat (~2GB)
echo ""
echo "installing CPU-only torch..."
pip install torch --index-url https://download.pytorch.org/whl/cpu

# install edgemind
echo ""
echo "installing EdgeMind..."
pip install -e .

echo ""
echo "========================================"
echo "✓ EdgeMind installed"
echo ""
echo "next steps:"
echo "  1. edit edgemind/core/config.py with your provider settings"
echo "  2. add your API key to .env file"
echo "  3. edgemind ingest data/docs"
echo "  4. edgemind interactive"
echo "========================================"
