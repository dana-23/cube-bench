from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import hydra
from omegaconf import DictConfig, OmegaConf

from .config import Config

ALL_TESTS = [
    "verification",
    "reconstruction",
    "prediction",
    "step_by_step",
    "learning_curve",
    "move_effect",
    "invariance_sweep",
    "reflection",
]


def setup_logging(verbosity: int, log_file: str | None = None) -> None:
    """0 = warnings/errors only; 1+ = info-level progress."""
    level = logging.INFO if verbosity >= 1 else logging.WARNING
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        handlers=handlers,
        force=True,
    )
    for noisy in ("matplotlib", "PIL", "urllib3", "httpx", "httpcore", "fontTools"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _load_test_configs_for_all() -> List[DictConfig]:
    """Load each test config in isolation for `test=all`."""
    base = Path(__file__).parent / "configs" / "test"
    return [OmegaConf.load(base / f"{t}.yaml") for t in ALL_TESTS]


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_logging(cfg.verbose, cfg.log_file)

    if cfg.build:
        # Preload optimal solver pruning tables and exit.
        import cube_bench.optimal.solver  # noqa: F401
        return

    pkg_prompts_dir = Path(__file__).parent / "prompts"
    prompts_path = Path(
        cfg.paths.prompts_path
    ) if cfg.paths.prompts_path else pkg_prompts_dir / "prompts.yaml"
    reflection_prompts = (
        Path(cfg.paths.reflection_prompts)
        if cfg.paths.reflection_prompts
        else pkg_prompts_dir / "reflection.yaml"
    )

    config = Config(
        dataset_path=Path(cfg.paths.dataset_path),
        prompts_path=prompts_path,
        results_dir=Path(cfg.paths.results_dir),
    )

    from .orchestrator import TestOrchestrator
    orch = TestOrchestrator(model_name=cfg.model, config=config, backend=cfg.backend)

    test_configs = (
        _load_test_configs_for_all() if cfg.test.name == "all" else [cfg.test]
    )

    if cfg.test.name == "all":
        print(f"🚀 Running ALL tests: {[c.name for c in test_configs]}")

    verbose_flag = cfg.verbose >= 1

    results_summary: dict[str, str] = {}
    try:
        for test_cfg in test_configs:
            tname = test_cfg.name
            print(f"\n=== ▶️ Starting Test: {tname} ===")
            try:
                orch.run_test(
                    test_cfg=test_cfg,
                    num_samples=cfg.samples,
                    verbose=verbose_flag,
                    reflection_prompts_default=reflection_prompts,
                )
                results_summary[tname] = "✅ Completed"
            except Exception as e:
                logging.exception(f"Test {tname} failed: {e}")
                results_summary[tname] = "❌ Failed"

        print("\n" + "=" * 40)
        print("📊 EXECUTION SUMMARY")
        print("=" * 40)
        for t, status in results_summary.items():
            print(f"{t:<20} {status}")
    finally:
        orch.cleanup()


if __name__ == "__main__":
    main()
