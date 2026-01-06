#!/bin/bash
# Release script for creating new versions

set -e

if [ -z "$1" ]; then
    echo "Usage: ./scripts/release.sh <version>"
    echo "Example: ./scripts/release.sh 0.2.0"
    exit 1
fi

VERSION=$1

echo "Preparing release $VERSION..."

# Run tests
echo "Running tests..."
pytest

# Run linters
echo "Running linters..."
ruff check src tests
black --check src tests
mypy src

# Update version in pyproject.toml
echo "Updating version in pyproject.toml..."
sed -i "s/version = \".*\"/version = \"$VERSION\"/" pyproject.toml

# Update version in __init__.py
echo "Updating version in __init__.py..."
sed -i "s/__version__ = \".*\"/__version__ = \"$VERSION\"/" src/taco/__init__.py

# Build package
echo "Building package..."
python -m build

# Create git tag
echo "Creating git tag v$VERSION..."
git add pyproject.toml src/taco/__init__.py
git commit -m "chore: bump version to $VERSION"
git tag -a "v$VERSION" -m "Release v$VERSION"

echo "Release $VERSION prepared!"
echo "Run 'git push && git push --tags' to publish."
