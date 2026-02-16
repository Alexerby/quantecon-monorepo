#!/bin/bash

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m virtualenv .venv
fi

# Activate and Install
source .venv/bin/activate
echo "Installing quantecon_lib and dependencies..."
pip install --upgrade pip
pip install -e .

# Run the analysis
echo "Running AAPL Bootstrap Analysis..."
python scripts/run_bootstrap_aapl.py
