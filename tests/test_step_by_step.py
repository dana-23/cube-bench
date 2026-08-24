"""Closed-loop scoring regressions for the step-by-step evaluation."""

# pylint: disable=missing-class-docstring,missing-function-docstring,protected-access

import json

from cube_bench.evaluations import step_by_step


class FakeFormula:
    def __init__(self, moves):
        self.moves = list(moves)

    def reverse(self):
        self.moves.reverse()
        return self

    def __str__(self):
        return " ".join(self.moves)


class BranchingCube:
    """Three-step state graph where an alternate first move stales the initial plan."""

    scramble_calls = []

    def __init__(self):
        self.state = 0

    def scramble(self, **kwargs):
        self.scramble_calls.append(kwargs)
        return FakeFormula(["TAIL", "STALE", "T0"])

    def is_solved(self):
        return self.state == 3

    def get_distance(self):
        return 3 - self.state

    def solve(self):
        return {
            0: "T0 STALE TAIL",
            1: "T1 T2",
            2: "T2",
            3: "",
        }[self.state]

    def apply(self, move):
        valid = {
            (0, "T0"): 1,
            (0, "ALT"): 1,
            (1, "T1"): 2,
            (2, "T2"): 3,
        }
        self.state = valid[(self.state, move)]

    def to_image(self):
        return None

    def __str__(self):
        return f"state-{self.state}"


class SequenceAssistant:
    def __init__(self):
        self.replies = iter((
            "<ANSWER> B </ANSWER>",
            "<ANSWER> A </ANSWER>",
            "<ANSWER> A </ANSWER>",
        ))

    def generate(self, **_kwargs):
        return next(self.replies)

    def get_name(self):
        return "sequence-assistant"


class ControlledStepByStep(step_by_step.StepByStepTest):
    GOOD_MOVES = {
        0: {"T0", "ALT"},
        1: {"T1"},
        2: {"T2"},
    }

    @classmethod
    def optimal_first_moves(cls, vc):
        return cls.GOOD_MOVES[vc.state]

    def gen_mcq_balanced(self, vc, teacher_move, rng, *, good_moves=None):
        del rng, good_moves
        if vc.state == 0:
            return {"A": "T0", "B": "ALT", "C": "BAD", "D": "WORSE"}, "A"
        return {"A": teacher_move, "B": "BAD", "C": "WORSE", "D": "NOPE"}, "A"


def test_oracle_gold_uses_letter_tiebreak_across_all_optimal_options():
    options = {"A": "BAD", "B": "ALT", "C": "TEACHER", "D": "WORSE"}
    assert step_by_step.StepByStepTest._oracle_gold_letter(
        options, {"ALT", "TEACHER"}
    ) == "B"


def test_episode_replans_after_alternate_optimal_move(monkeypatch):
    BranchingCube.scramble_calls = []
    monkeypatch.setattr(step_by_step, "VirtualCube", BranchingCube)
    task = ControlledStepByStep(
        SequenceAssistant(),
        config=None,
        n_moves=3,
        checkpoint=False,
    )

    result = task._run_episode(19)
    steps = result["sample_log"]["steps_data"]

    assert BranchingCube.scramble_calls == [
        {"random_seed": 19, "n_moves": 3, "exact_depth": True}
    ]
    assert [step["teacher_move"] for step in steps] == ["T0", "T1", "T2"]
    assert [step["oracle_distance"] for step in steps] == [3, 2, 1]
    assert steps[0]["is_prompt_correct"] is False
    assert all(step["is_correct"] for step in steps)
    assert result["correct_steps"] == 3
    assert result["perfect_solve"] is True
    assert result["first_error_step"] is None
    assert result["sample_log"]["final_solved"] is True


def test_checkpoint_loader_ignores_legacy_records(tmp_path):
    legacy = {"sample_id": 0, "result": {"correct_steps": 3}}
    current = {"sample_id": 1, "result": {"correct_steps": 3, "perfect_solve": True}}
    path = tmp_path / "checkpoint.jsonl"
    path.write_text(f"{json.dumps(legacy)}\n{json.dumps(current)}\n", encoding="utf-8")

    assert step_by_step.StepByStepTest._load_checkpoint(path) == {1: current["result"]}
