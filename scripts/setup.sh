#!/bin/bash
# Setup script for development environment

set -e

echo "Setting up TACO development environment..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install package with dev dependencies
echo "Installing package with dev dependencies..."
pip install -e ".[dev,docs]"

# Install pre-commit hooks
echo "Installing pre-commit hooks..."
pre-commit install

# Run initial checks
echo "Running initial checks..."
ruff check src tests
black --check src tests
mypy src

echo "Setup complete! Run 'source venv/bin/activate' to activate the environment."
