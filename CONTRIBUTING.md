# Contributing to TACO

Thank you for considering contributing to TACO! This document provides guidelines and instructions for contributing.

## Development Setup

1. Fork and clone the repository:
```bash
git clone https://github.com/yourusername/taco.git
cd taco
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install development dependencies:
```bash
pip install -e ".[dev,docs]"
```

4. Install pre-commit hooks:
```bash
pre-commit install
```

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Your Changes

- Write clear, concise code
- Follow the code style guidelines
- Add tests for new functionality
- Update documentation as needed

### 3. Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=taco --cov-report=html

# Run specific tests
pytest tests/test_version.py
```

### 4. Check Code Quality

```bash
# Format code
black src tests

# Lint code
ruff check src tests

# Type checking
mypy src

# Or run all pre-commit hooks
pre-commit run --all-files
```

### 5. Commit Your Changes

```bash
git add .
git commit -m "feat: add amazing feature"
```

Follow conventional commit format:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `test:` - Test changes
- `refactor:` - Code refactoring
- `style:` - Code style changes
- `chore:` - Build/tooling changes

### 6. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub.

## Code Style Guidelines

### Python Style
- Follow PEP 8
- Maximum line length: 100 characters
- Use type hints for all functions
- Write docstrings (Google style) for all public APIs

### Example:
```python
def calculate_sum(a: int, b: int) -> int:
    """Calculate the sum of two numbers.

    Args:
        a: First number.
        b: Second number.

    Returns:
        The sum of a and b.

    Raises:
        TypeError: If inputs are not integers.
    """
    return a + b
```

## Testing Guidelines

- Write tests for all new features
- Maintain or improve code coverage
- Use descriptive test names
- Use fixtures for common test data

### Example:
```python
def test_calculate_sum_positive_numbers():
    """Test sum calculation with positive numbers."""
    result = calculate_sum(2, 3)
    assert result == 5
```

## Documentation

- Update documentation for any user-facing changes
- Add docstrings to all public functions and classes
- Update README.md if needed
- Add examples for new features

## Pull Request Process

1. Ensure all tests pass
2. Update documentation
3. Add entry to CHANGELOG.md
4. Request review from maintainers
5. Address review comments
6. Wait for approval and merge

## Questions?

Feel free to open an issue for:
- Bug reports
- Feature requests
- Questions about contributing
- General discussions

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Prioritize the community's best interests

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
