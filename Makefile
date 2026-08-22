# Cube Bench — task runner
# Assumes a Python environment (venv/conda) is already active.

PYTHON ?= python
PIP    ?= $(PYTHON) -m pip

.PHONY: help install install-dev build setup check lint test

help:
	@echo "Cube Bench targets:"
	@echo "  make install      - Install pinned deps and the package in editable mode"
	@echo "  make install-dev  - Install the package with lint and test tools"
	@echo "  make build        - Precompute IDA* / optimal-distance graphs (~8h)"
	@echo "  make setup        - install + build"
	@echo "  make check        - Verify the package imports"
	@echo "  make lint         - Run Pylint recursively over the repository"
	@echo "  make test         - Run the pytest suite"

install:
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

install-dev:
	$(PIP) install -r requirements.txt
	$(PIP) install -e ".[dev]"

build:
	cube-bench build=True

setup: install build

check:
	$(PYTHON) -c "import cube_bench as cb; print('cube_bench version:', getattr(cb, '__version__', 'unknown'))"

lint:
	$(PYTHON) -m pylint --recursive=y .

test:
	$(PYTHON) -m pytest
