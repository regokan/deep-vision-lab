.PHONY: install format

# Install dependencies
install:
	poetry install
# Formatting Python code with isort and black using Poetry
format:
	poetry run python -m isort .
	poetry run python -m black .
test:
	PYTHONPATH=./ poetry run python -m pytest -vv
setup-ubuntu:
	sudo apt update
	sudo apt install -y curl python3-pip python3-venv
	curl -sSL https://install.python-poetry.org | python3 -
	echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
	source ~/.bashrc
	poetry --version
	sudo growpart /dev/nvme0n1 1
	sudo resize2fs /dev/nvme0n1p1
