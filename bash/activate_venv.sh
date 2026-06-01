#!/usr/bin/env bash

VENV_DIR=".venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "Virtual environment not found: $VENV_DIR"
    echo "Run first:"
    echo "bash scripts/setup_venv.sh"
    return 1 2>/dev/null || exit 1
fi

echo "Activating virtual environment: $VENV_DIR"
source "$VENV_DIR/bin/activate"

echo "Activated"
echo "Python: $(which python)"
echo "Pip:    $(which pip)"