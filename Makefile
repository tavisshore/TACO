.PHONY: help install install-dev test lint format type-check clean docs build

help:
	@echo "Available commands:"
	@echo "  install       Install package"
	@echo "  install-dev   Install package with dev dependencies"
	@echo "  test          Run tests"
	@echo "  lint          Run linters"
	@echo "  format        Format code"
	@echo "  type-check    Run type checking"
	@echo "  clean         Clean build artifacts"
	@echo "  docs          Build documentation"
	@echo "  build         Build package"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev,docs]"
	pre-commit install

test:
	pytest

test-cov:
	pytest --cov=taco --cov-report=html --cov-report=term

lint:
	ruff check src tests

format:
	black src tests
	ruff check --fix src tests

type-check:
	mypy src

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

docs:
	cd docs && make html

build:
	python -m build

publish-test:
	python -m build
	twine upload --repository testpypi dist/*

publish:
	python -m build
	twine upload dist/*
