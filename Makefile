.PHONY: install format

# Install dependencies
install:
	poetry install
# Formatting Python code with isort and black using Poetry
format:
	poetry run python -m isort .
	poetry run python -m black .
