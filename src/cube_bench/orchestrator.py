from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any, Optional

from omegaconf import DictConfig

from .config import Config
from .tests import (
    InvarianceSweepTest,
    LearningCurveTest,
    MoveEffectTest,
    ReconstructionTest,
    ReflectionTest,
    SolveMovesTest,
    StepByStepTest,
    VerificationTest,
)
from cube_bench.runtime.model_assistant import ModelAssistant

logger = logging.getLogger(__name__)


class TestOrchestrator:
    def __init__(self, model_name: str, config: Config, backend: str = "hf"):
        logger.info("Initializing assistant backend=%s engine=%s", model_name, backend)
        self.assistant = ModelAssistant(model_name, engine=backend)
        self.config = config

    def run_test(
        self,
        test_cfg: DictConfig,
        num_samples: int,
        verbose: bool = False,
        reflection_prompts_default: Optional[Path] = None,
    ) -> Any:
        name = test_cfg.name

        if name == "prediction":
            test = SolveMovesTest(
                self.assistant, self.config,
                prompt_type=test_cfg.prompt_type,
                n_moves=test_cfg.n_moves,
                verbose=verbose,
            )

        elif name == "verification":
            test = VerificationTest(
                self.assistant, self.config,
                n_moves=test_cfg.n_moves,
                verbose=verbose,
            )

        elif name == "reconstruction":
            test = ReconstructionTest(
                self.assistant, self.config,
                n_moves=test_cfg.n_moves,
                verbose=verbose,
            )

        elif name == "step-by-step":
            test = StepByStepTest(
                self.assistant, self.config,
                n_moves=test_cfg.n_moves,
                verbose=verbose,
                idk_enabled=test_cfg.idk_enabled,
                idk_weight=test_cfg.idk_weight,
                idk_policy=test_cfg.idk_policy,
                idk_conf_threshold=test_cfg.idk_conf_threshold,
            )

        elif name == "learning-curve":
            test = LearningCurveTest(
                self.assistant, self.config,
                n_moves=test_cfg.n_moves,
                max_attempts=test_cfg.max_attempts,
                accept_progress=test_cfg.accept_progress,
                verbose=verbose,
            )

        elif name == "move-effect":
            test = MoveEffectTest(
                self.assistant, self.config,
                n_moves=test_cfg.n_moves,
                verbose=verbose,
            )

        elif name == "invariance-sweep":
            test = InvarianceSweepTest(
                self.assistant, self.config,
                n_moves=test_cfg.n_moves,
                verbose=verbose,
                balance_gold_letters=test_cfg.balance_gold_letters,
                add_labels=test_cfg.add_labels,
                max_new_tokens=test_cfg.max_new_tokens,
            )

        elif name == "reflection":
            rp = (
                Path(test_cfg.reflection_prompts)
                if test_cfg.get("reflection_prompts")
                else (reflection_prompts_default
                      or (self.config.prompts_path.parent / "reflection.yaml"))
            )
            test = ReflectionTest(
                assistant=self.assistant,
                config=self.config,
                reflection_prompts=rp,
                reflection_type=test_cfg.reflection_type,
                prompt_type=test_cfg.prompt_type,
                max_reflections=test_cfg.max_reflections,
                verbose=verbose,
            )

        else:
            raise ValueError(f"Unknown test name: {name!r}")

        return test.run(num_samples)

    def cleanup(self):
        logger.info("Cleaning up assistant resources…")
        if hasattr(self, "assistant") and self.assistant is not None:
            self.assistant.cleanup()
            del self.assistant
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
            logger.info("Assistant resources cleaned up.")
