# Cube Bench

> A reproducible Rubik's Cube benchmark suite for probing multimodal LLMs across perception, grounding, and closed-loop control.

**Cube Bench** is a framework designed to evaluate the reasoning and planning capabilities of Large Multimodal Models (LMMs) using the Rubik's Cube as a complex, structured environment. It focuses on three core pillars:

1. **Perception:** Can the model accurately recognize the state of the cube from images?
2. **Grounding:** Can the model map visual states to internal representations?
3. **Closed-Loop Control:** Can the model generate valid moves to reach a target state (e.g., solving the cube)?

---

## Install & Quick Start

### 1) Clone the repository

```bash
git clone <repo-url> cube-bench
cd cube-bench
```

---

### 2) Create & activate an environment (choose one)

**Option A: Conda (recommended for AI/ML)**

```bash
conda create -n cube_bench python=3.12
conda activate cube_bench
```

**Option B: Python venv**

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

---

### 3) Install & build

You have two options - a one-shot `make` flow or the explicit `pip` commands.

**Option A: Makefile (recommended)**

```bash
make setup    # = make install + make build
make check    # verify the install
```

Individual targets are also available: `make install`, `make build`, `make check`. Run `make help` to list them.

**Option B: Manual**

```bash
# 3a) Install pinned deps FIRST (prevents version conflicts)
pip install -r requirements.txt

# 3b) Install the package in editable dev mode
pip install -e .

# 3c) Precompute IDA* / optimal-distance graphs
# Warning: computationally intensive, can take ~8 hours depending on your CPU.
cube-bench --build

# 3d) Check the install
python -c "import cube_bench as cb; print('cube_bench version:', getattr(cb, '__version__', 'unknown'))"
```

> **Note:** The Makefile assumes your environment from step 2 is already active. It does not create or manage envs.
