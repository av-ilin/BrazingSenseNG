#!/usr/bin/env bash

set -e

VENV_DIR=".venv"
PYTHON_BIN="python3"

echo "BrazingSense environment setup"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Error: python3 not found"
    exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment: $VENV_DIR"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
else
    echo "Virtual environment already exists: $VENV_DIR"
fi

echo "Activating virtual environment"
source "$VENV_DIR/bin/activate"

echo "Upgrading pip"
python -m pip install --upgrade pip setuptools wheel

if [ ! -f "requirements.txt" ]; then
    echo "Error: requirements.txt not found"
    exit 1
fi

echo "Installing dependencies"
pip install -r requirements.txt

# echo "Installing project kernel for Jupyter"
# python -m ipykernel install --user --name brazing-sense --display-name "Python (BrazingSense)"

echo "Done"
echo ""
echo "To activate the environment, run:"
echo "source .venv/bin/activate"