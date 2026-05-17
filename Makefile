# Cube Bench — task runner
# Assumes a Python environment (venv/conda) is already active.

PYTHON ?= python
PIP    ?= $(PYTHON) -m pip

.PHONY: help install build setup check

help:
	@echo "Cube Bench targets:"
	@echo "  make install  - Install pinned deps and the package in editable mode"
	@echo "  make build    - Precompute IDA* / optimal-distance graphs (~8h)"
	@echo "  make setup    - install + build"
	@echo "  make check    - Verify the package imports"

install:
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

build:
	cube-bench --build

setup: install build

check:
	$(PYTHON) -c "import cube_bench as cb; print('cube_bench version:', getattr(cb, '__version__', 'unknown'))"
