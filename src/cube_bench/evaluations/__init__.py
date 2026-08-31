"""The benchmark's evaluation tasks, re-exported for the orchestrator."""

from .solve_moves import SolveMovesTest
from .verification import VerificationTest
from .reconstruction import ReconstructionTest
from .step_by_step import StepByStepTest
from .learning_curve import LearningCurveTest
from .move_effect import MoveEffectTest
from .reflection import ReflectionTest

__all__ = [
    "SolveMovesTest",
    "VerificationTest",
    "ReconstructionTest",
    "StepByStepTest",
    "LearningCurveTest",
    "MoveEffectTest",
    "ReflectionTest"
]
