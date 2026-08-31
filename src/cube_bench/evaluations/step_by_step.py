"""Step-by-step task: closed-loop solving, one move per turn."""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from cube_bench.core import scoring
from cube_bench.core import ItemRecord, BaseTest
from cube_bench.core.results import append_jsonl, read_checkpoint
from cube_bench.prompts.prompt_factory import PromptFactory
from cube_bench.sim.cube_simulator import VirtualCube

logger = logging.getLogger(__name__)


@dataclass
class _StepContext:
    """Oracle state, prompt, and replay data for one closed-loop decision."""

    episode_idx: int
    step_idx: int
    teacher_plan: List[str]
    teacher_move: str
    oracle_distance: int
    good_moves: set[str]
    state_text: str
    state_image: Any
    options: Dict[str, str]
    gold_letter: str
    system_prompt: str
    user_prompt: str
    replay: bool
    seed_steps: List[Dict[str, Any]]


@dataclass
class _StepPrediction:
    """Parsed response and optional latency for one decision."""

    response: str
    letter: Optional[str]
    prompt_correct: bool
    latency: Optional[float] = None


@dataclass
class _StepOutcome:
    """Episode-local effects produced by resolving one prediction."""

    step_log: Dict[str, Any]
    teacher_plan: List[str]
    continue_episode: bool
    oracle_correct: bool = False
    teacher_help: bool = False
    confusion_pair: Optional[Tuple[str, str]] = None
    latency: Optional[float] = None


@dataclass
class _RunTotals:
    """Episode results merged in seed order, independent of execution order."""

    per_step_totals: List[int] = field(default_factory=list)
    first_error_step: List[int] = field(default_factory=list)
    confusion: Dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    solve_depths: List[int] = field(default_factory=list)
    perfect_flags: List[bool] = field(default_factory=list)
    sample_logs: List[Dict[str, Any]] = field(default_factory=list)
    missing: List[int] = field(default_factory=list)
    completed: int = 0

    @classmethod
    def for_depth(cls, n_moves: int) -> "_RunTotals":
        """Totals with the per-step vectors sized for a ``n_moves``-deep episode."""
        return cls(per_step_totals=[0] * n_moves)


class StepByStepTest(BaseTest):
    """Closed-loop: ask for next move at each step, accept optimal or teacher."""

    test_type = "step_by_step"

    # ----- Construction -----

    def __init__(  # pylint: disable=unused-argument  # idk_conf_threshold is pinned below
        self,
        assistant,
        config,
        *,
        n_moves: int = 8,
        verbose: bool = False,
        idk_enabled: bool = False,
        idk_weight: float = 0.25,
        idk_policy: str = "teacher_on_abstain",
        idk_conf_threshold: Optional[float] = None,
        history_enabled: bool = False,
        history_images: bool = True,
        history_full_responses: bool = True,
        concurrency: int = 1,
        checkpoint: bool = True,
        checkpoint_dir: Optional[str] = None,
        seed_run: Optional[str] = None,
        seed_prefix: int = 1,
    ):
        super().__init__(assistant, config, n_moves, verbose)

        self._totals = _RunTotals.for_depth(n_moves)

        self.idk_enabled = bool(idk_enabled)
        self.idk_weight = float(idk_weight)
        self.idk_policy = str(idk_policy)
        # NOTE: the `idk_conf_threshold` argument is deliberately ignored — this
        # threshold stays pinned at 50 to keep results comparable with earlier runs.
        self.idk_conf_threshold = 50

        # The history arm sees prior state/action turns; the Markov arm sees only
        # the current state.
        self.history_enabled = bool(history_enabled)
        self.history_images = bool(history_images)
        self.history_full_responses = bool(history_full_responses)

        # Episodes may run concurrently, but their closed-loop steps remain serial.
        # Local HF/vLLM models should use one worker.
        self.concurrency = max(1, int(concurrency))
        thinking_budget = getattr(config, "thinking_budget", None)
        self.thinking_budget = None if thinking_budget is None else int(thinking_budget)

        # A config-keyed JSONL sidecar makes completed episodes resumable.
        self.checkpoint = bool(checkpoint)
        self.checkpoint_dir = checkpoint_dir

        # Replay is valid only through the prefix where both arms had identical
        # context; for Markov versus history, that is normally the first step.
        self.seed_prefix = max(0, int(seed_prefix))
        self._seed_steps: Dict[int, List[Dict[str, Any]]] = {}
        self.seed_run = seed_run
        if seed_run and self.seed_prefix > 0:
            self._seed_steps = self._load_seed_run(Path(seed_run))
            logger.info(
                "Seeding first %d step(s) per episode from %s (%d episodes) — arm=%s",
                self.seed_prefix, seed_run, len(self._seed_steps),
                "history" if history_enabled else "markov",
            )

        if self.idk_enabled:
            logger.info("IDK Enabled -- Policy: %s -- Weight: %s", self.idk_policy, self.idk_weight)
        logger.info(
            "Arm: %s%s",
            "history" if self.history_enabled else "markov",
            (f" (images={self.history_images}, full_responses={self.history_full_responses})"
             if self.history_enabled else ""),
        )

    def _eval(self, text: str, gold_letter: str) -> Tuple[bool, Optional[str]]:
        """Parse model output; return (is_correct, predicted_letter or 'IDK'/None)."""
        if self.idk_enabled and self.parse_idk(text):
            return False, "IDK"
        pred = self.parse_letter(text)
        if pred is None:
            logger.warning("Could not parse model's output")
            logger.info("Model's full response:\n\n%s", text)
            return False, None
        return (pred == gold_letter), pred

    def _build_prompts(self, kwargs: Dict[str, Any]) -> Tuple[str, str]:
        """Build standard prompts or the abstention-aware variant."""
        if not self.idk_enabled:
            return PromptFactory.get("step_by_step", **kwargs)

        n_moves = kwargs.get("n_moves", self.n_moves)
        text_state = kwargs.get("textual_representation", "")
        A, B, C, D = kwargs["move_A"], kwargs["move_B"], kwargs["move_C"], kwargs["move_D"]
        conf_thr = self.idk_conf_threshold

        sys_prompt = (
            "You are an expert Rubik's-Cube solver. Pick exactly ONE move (A, B, C, D or IDK) "
            "that most reduces the cube's distance to solved.\n\n"
            "Context\n"
            f"- The cube is {n_moves} moves from solved under God's distance (HTM).\n"
            "- You will receive a textual cube state (ground truth) and an image (reference only). "
            "Use the TEXT ONLY to decide.\n\n"
            "Decision rule (deterministic)\n"
            "1) For each candidate, internally simulate that move on the textual state "
            "and estimate the resulting distance d1 (HTM).\n"
            "2) If any candidate solves the cube (d1=0), choose that candidate.\n"
            "3) Otherwise choose the candidate with the lowest d1.\n"
            "4) If there is a tie on d1, break ties by letter: A ≺ B ≺ C ≺ D.\n"
        )
        if conf_thr is not None:
            sys_prompt += f"5) If you are <{int(conf_thr)}% confident, abstain with IDK.\n"

        sys_prompt += (
            "\nOutput format (STRICT)\n"
            "Return exactly one of the following on a single line:\n"
            "<ANSWER> A </ANSWER>\n"
            "<ANSWER> B </ANSWER>\n"
            "<ANSWER> C </ANSWER>\n"
            "<ANSWER> D </ANSWER>\n"
            "<ANSWER> IDK </ANSWER>\n\n"
            "No explanations, no extra text.\n"
        )

        user_prompt = (
            f"The cube is {n_moves} moves away from being solved.\n\n"
            "Textual Cube State (ground truth):\n"
            f"{text_state}\n\n"
            "Candidate moves:\n"
            f"A: {A}\nB: {B}\nC: {C}\nD: {D}\n"
            "E: I don't know (abstain)\n\n"
            "Respond with exactly one line (A/B/C/D or IDK):\n"
            "<ANSWER> X </ANSWER>"
        )
        return sys_prompt, user_prompt

    @staticmethod
    def _oracle_gold_letter(options: Dict[str, str], good_moves: set[str]) -> str:
        """Return the letter-tiebroken option among moves that reduce oracle distance."""
        candidates = [letter for letter, move in options.items() if move in good_moves]
        if not candidates:
            raise RuntimeError("MCQ contains no oracle-optimal option")
        return min(candidates)

    @staticmethod
    def _refresh_teacher_plan(cube: VirtualCube, plan: List[str], good_moves: set[str]) -> List[str]:
        """Keep a valid plan or replace a stale plan from the state actually reached."""
        if plan and plan[0] in good_moves:
            return plan
        refreshed = cube.solve().split()
        if not refreshed or refreshed[0] not in good_moves:
            raise RuntimeError("Oracle solver did not return a progress-making teacher move")
        return refreshed

    def _prepare_step(
        self,
        cube: VirtualCube,
        teacher_plan: List[str],
        episode_idx: int,
        step_idx: int,
    ) -> _StepContext:
        """Build the oracle labels, MCQ, prompt, and replay metadata for one step."""
        oracle_distance = cube.get_distance()
        good_moves = self.optimal_first_moves(cube)
        teacher_plan = self._refresh_teacher_plan(cube, teacher_plan, good_moves)
        teacher_move = teacher_plan[0]
        state_text = self.state_text(cube)
        state_image = cube.to_image()

        step_rng = self.item_rng(self.n_moves, episode_idx, step_idx)
        options, _teacher_letter = self.gen_mcq_balanced(
            cube,
            teacher_move,
            step_rng,
            good_moves=good_moves,
        )
        gold_letter = self._oracle_gold_letter(options, good_moves)
        system_prompt, user_prompt = self._build_prompts({
            "n_moves": oracle_distance,
            "n_moves_optimal": max(oracle_distance - 1, 0),
            "textual_representation": state_text,
            "move_A": options["A"],
            "move_B": options["B"],
            "move_C": options["C"],
            "move_D": options["D"],
            "metric": "HTM (Half-Turn Metric)",
        })

        seed_steps = self._seed_steps.get(episode_idx, [])
        replay = (
            step_idx < self.seed_prefix
            and step_idx < len(seed_steps)
        )
        return _StepContext(
            episode_idx=episode_idx,
            step_idx=step_idx,
            teacher_plan=teacher_plan,
            teacher_move=teacher_move,
            oracle_distance=oracle_distance,
            good_moves=good_moves,
            state_text=state_text,
            state_image=state_image,
            options=options,
            gold_letter=gold_letter,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            replay=replay,
            seed_steps=seed_steps,
        )

    def _history_payload(self, convo: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
        """Return prior turns in the configured history representation."""
        if not self.history_enabled or not convo:
            return None
        if self.history_images:
            return list(convo)
        return [{key: value for key, value in turn.items() if key != "image"} for turn in convo]

    def _predict_step(
        self,
        context: _StepContext,
        convo: List[Dict[str, Any]],
    ) -> _StepPrediction:
        """Replay a seeded response or query the assistant for the current step."""
        if context.replay:
            seed = context.seed_steps[context.step_idx]
            if context.state_text != seed["cube_state"]:
                raise AssertionError(
                    f"[ep {context.episode_idx} step {context.step_idx}] "
                    "seed state mismatch (scramble/seeds drifted)"
                )
            if context.options != seed["options"] or context.gold_letter != seed["correct_letter"]:
                raise AssertionError(
                    f"[ep {context.episode_idx} step {context.step_idx}] "
                    "seed MCQ mismatch (MCQ generation drifted)"
                )
            response = seed.get("full_response", "")
            letter = seed.get("predicted_letter")
            return _StepPrediction(response, letter, letter == context.gold_letter)

        started = time.time()
        response = self.ask(
            user_prompt=context.user_prompt,
            system_prompt=context.system_prompt,
            image=context.state_image,
            history=self._history_payload(convo),
            track_latency=False,
        )
        latency = time.time() - started
        prompt_correct, letter = self._eval(response, context.gold_letter)
        return _StepPrediction(response, letter, prompt_correct, latency)

    def _append_history(
        self,
        convo: List[Dict[str, Any]],
        context: _StepContext,
        prediction: _StepPrediction,
    ) -> None:
        """Append the current turn to the history-arm transcript."""
        if not self.history_enabled:
            return
        convo.append({"role": "user", "text": context.user_prompt, "image": context.state_image})
        if self.history_full_responses or not prediction.letter or prediction.letter == "IDK":
            assistant_text = prediction.response
        else:
            assistant_text = f"<ANSWER> {prediction.letter} </ANSWER>"
        convo.append({"role": "assistant", "text": assistant_text})

    @staticmethod
    def _step_log(
        context: _StepContext,
        prediction: _StepPrediction,
        **extra: Any,
    ) -> Dict[str, Any]:
        """Build the shared per-step log fields and merge outcome-specific data."""
        record = {
            "step": context.step_idx,
            "cube_state": context.state_text,
            "options": context.options,
            "correct_letter": context.gold_letter,
            "teacher_move": context.teacher_move,
            "oracle_good_moves": sorted(context.good_moves),
            "oracle_distance": context.oracle_distance,
            "full_response": prediction.response,
            "predicted_letter": prediction.letter,
        }
        record.update(extra)
        return record

    def _validate_replayed_trajectory(self, cube: VirtualCube, context: _StepContext) -> None:
        """Ensure an applied replay move reconstructs the seed run's next state."""
        if not context.replay or context.step_idx + 1 >= len(context.seed_steps):
            return
        rebuilt = self.state_text(cube)
        expected = context.seed_steps[context.step_idx + 1]["cube_state"]
        if rebuilt != expected:
            raise AssertionError(
                f"[ep {context.episode_idx}] rebuilt S{context.step_idx + 1} "
                "!= seed's logged state (replayed move application diverged)"
            )

    def _resolve_step(
        self,
        cube: VirtualCube,
        context: _StepContext,
        prediction: _StepPrediction,
    ) -> _StepOutcome:
        """Score and apply one prediction, returning its episode-local effects."""
        if prediction.letter is None:
            if self.verbose:
                logger.info(
                    "[sample %s] Parse failure at step %s; ending episode.",
                    context.episode_idx,
                    context.step_idx + 1,
                )
            return _StepOutcome(
                step_log=self._step_log(
                    context,
                    prediction,
                    is_correct=False,
                    is_prompt_correct=False,
                    parse_fail=True,
                ),
                teacher_plan=context.teacher_plan,
                continue_episode=False,
                latency=prediction.latency,
            )

        if self.idk_enabled and prediction.letter == "IDK":
            logger.info("Model responded with IDK.")
            teacher_help = self.idk_policy == "teacher_on_abstain"
            if teacher_help:
                cube.apply(context.teacher_move)
                context.teacher_plan.pop(0)
            return _StepOutcome(
                step_log=self._step_log(
                    context,
                    prediction,
                    is_correct=False,
                    is_prompt_correct=False,
                    abstained=True,
                    idk_policy=self.idk_policy,
                ),
                teacher_plan=context.teacher_plan,
                continue_episode=teacher_help,
                teacher_help=teacher_help,
                latency=prediction.latency,
            )

        chosen_move = context.options.get(prediction.letter)
        oracle_correct = bool(chosen_move in context.good_moves) if chosen_move else False
        if self.verbose:
            logger.info("Model's chosen option: %s -> %s", prediction.letter, chosen_move)
            logger.info(
                "[Sample: %s Step: %s] Oracle-good moves: %s",
                context.episode_idx,
                context.step_idx,
                sorted(context.good_moves),
            )

        teacher_plan = context.teacher_plan
        if chosen_move and oracle_correct:
            cube.apply(chosen_move)
            if chosen_move == context.teacher_move:
                teacher_plan.pop(0)
            else:
                teacher_plan = []
            self._validate_replayed_trajectory(cube, context)
        elif self.verbose:
            logger.info(
                "[sample %s] First error at step %s",
                context.episode_idx,
                context.step_idx + 1,
            )

        return _StepOutcome(
            step_log=self._step_log(
                context,
                prediction,
                chosen_move=chosen_move,
                is_correct=oracle_correct,
                is_prompt_correct=prediction.prompt_correct,
            ),
            teacher_plan=teacher_plan,
            continue_episode=oracle_correct,
            oracle_correct=oracle_correct,
            confusion_pair=(context.teacher_move, chosen_move) if chosen_move else None,
            latency=prediction.latency,
        )

    def _run_step(
        self,
        cube: VirtualCube,
        teacher_plan: List[str],
        episode_idx: int,
        step_idx: int,
        convo: List[Dict[str, Any]],
    ) -> _StepOutcome:
        """Prepare, predict, record, and resolve one closed-loop step."""
        context = self._prepare_step(cube, teacher_plan, episode_idx, step_idx)
        prediction = self._predict_step(context, convo)
        self._append_history(convo, context, prediction)
        return self._resolve_step(cube, context, prediction)

    def _run_episode(self, idx: int) -> Dict[str, Any]:
        """Run a serial episode with thread-safe, episode-local counters."""
        cube = VirtualCube()
        scramble = cube.scramble(random_seed=idx, n_moves=self.n_moves, exact_depth=True)
        initial_solution_path = self.teacher_path(scramble)
        teacher_plan = list(initial_solution_path)
        correct_steps = 0
        teacher_help = 0

        if self.verbose:
            logger.info("[sample %s] Scramble: %s", idx, scramble)
            logger.info("[sample %s] Initial teacher path: %s", idx, initial_solution_path)

        sample_log = {
            "sample_id": idx,
            "scramble": str(scramble),
            "solution_path": initial_solution_path,
            "steps_data": [],
        }

        # Episode transcript for the history arm: alternating turns per prior step.
        convo: List[Dict[str, Any]] = []
        per_step_totals = [0] * self.n_moves
        confusion_pairs: List[Tuple[str, str]] = []
        latencies: List[float] = []
        first_error_step: Optional[int] = None

        for step_i in range(self.n_moves):
            if cube.is_solved():
                break

            outcome = self._run_step(cube, teacher_plan, idx, step_i, convo)
            teacher_plan = outcome.teacher_plan
            sample_log["steps_data"].append(outcome.step_log)
            per_step_totals[step_i] += 1
            correct_steps += int(outcome.oracle_correct)
            teacher_help += int(outcome.teacher_help)
            if outcome.confusion_pair:
                confusion_pairs.append(outcome.confusion_pair)
            if outcome.latency is not None:
                latencies.append(outcome.latency)
            if not outcome.continue_episode:
                first_error_step = step_i + 1
                break

        solved = cube.is_solved()
        sample_log["final_solved"] = solved
        sample_log["oracle_correct_steps"] = correct_steps

        return {
            "sample_log": sample_log,
            "correct_steps": correct_steps,
            "perfect_solve": solved and correct_steps == self.n_moves and teacher_help == 0,
            "per_step_totals": per_step_totals,
            "confusion_pairs": confusion_pairs,
            "latencies": latencies,
            "first_error_step": first_error_step,
        }

    def _load_seed_run(self, path: Path) -> Dict[int, List[Dict[str, Any]]]:
        """Load a prior run's per-episode step logs, keyed by sample_id."""
        payload = json.loads(path.read_text(encoding="utf-8"))
        p = payload[-1] if isinstance(payload, list) else payload
        if "samples" not in p:
            raise ValueError(
                f"Seed run {path} has no per-sample logs (predates that feature); "
                "cannot replay from it."
            )
        if p.get("n_moves_scrambled") != self.n_moves:
            raise ValueError(
                f"Seed run depth d={p.get('n_moves_scrambled')} != current d={self.n_moves}."
            )
        seed_model = p.get("model_name")
        if seed_model != self.assistant.get_name():
            raise ValueError(
                f"Seed run model '{seed_model}' != current model "
                f"'{self.assistant.get_name()}'. Replaying another model's decisions "
                "would not be a valid same-model comparison."
            )
        return {s["sample_id"]: s["steps_data"] for s in p["samples"]}

    def _checkpoint_path(self, num_samples: int) -> Optional[Path]:
        """Return the stable, config-keyed checkpoint path outside the run directory."""
        if not self.checkpoint:
            return None
        base = (
            Path(self.checkpoint_dir)
            if self.checkpoint_dir
            else Path.cwd() / "checkpoints" / "step_by_step" / date.today().isoformat()
        )
        arm = "history" if self.history_enabled else "markov"
        thinking_tag = f"_tb{self.thinking_budget}" if self.thinking_budget is not None else ""
        # Keep seeded branches separate from plain-run checkpoints.
        seed_tag = f"_seeded-{Path(self.seed_run).stem}-p{self.seed_prefix}" if self._seed_steps else ""
        key = f"{self.assistant.get_name()}_{arm}_d{self.n_moves}_n{num_samples}{thinking_tag}{seed_tag}"
        safe = "".join(c if (c.isalnum() or c in "-._") else "_" for c in key)
        base.mkdir(parents=True, exist_ok=True)
        return base / f"{safe}.jsonl"

    @staticmethod
    def _load_checkpoint(path: Path) -> Dict[int, Dict[str, Any]]:
        return read_checkpoint(path, "perfect_solve")

    def desc(self) -> str:
        return f"Step-by-step ({self.n_moves} moves)"

    # ----- Execution -----

    def run_item(self, idx: int) -> ItemRecord:
        """Run one episode; ``extra['result']`` is the persisted checkpoint record."""
        return ItemRecord(index=idx, extra={"result": self._run_episode(idx)})

    def collect(self, num_samples: int) -> List[ItemRecord]:
        """Resume finished episodes from the checkpoint and run the rest."""
        ckpt_path = self._checkpoint_path(num_samples)
        results: Dict[int, Dict[str, Any]] = {}
        if ckpt_path is not None:
            results = self._load_checkpoint(ckpt_path)
            if results:
                self._logger.info("Resuming from %s - %d/%d episodes already done",
                                  ckpt_path, len(results), num_samples)

        todo = [idx for idx in range(num_samples) if idx not in results]
        ckpt_lock = threading.Lock()

        def _record(idx: int, record: ItemRecord) -> None:
            results[idx] = record.extra["result"]
            if ckpt_path is not None:
                with ckpt_lock:
                    append_jsonl(ckpt_path, {"sample_id": idx, "result": results[idx]})

        # Failed episodes remain absent from the checkpoint so a later run retries them.
        if todo:
            if self.concurrency > 1:
                self._logger.info(
                    "Running %d episodes with concurrency=%d", len(todo), self.concurrency)
                with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
                    futures = {pool.submit(self.run_item, idx): idx for idx in todo}
                    for fut in tqdm(as_completed(futures), total=len(todo), desc=self.desc()):
                        idx = futures[fut]
                        try:
                            _record(idx, fut.result())
                        except Exception as e:
                            self._logger.error("Episode %d failed permanently: %s", idx, e)
            else:
                for idx in tqdm(todo, desc=self.desc()):
                    try:
                        _record(idx, self.run_item(idx))
                    except Exception as e:
                        self._logger.error("Episode %d failed permanently: %s", idx, e)

        done = sorted(results)
        self._totals.missing = [idx for idx in range(num_samples) if idx not in results]
        if self._totals.missing:
            self._logger.warning(
                "%d/%d episodes incomplete: %s - re-run the same command to resume%s.",
                len(self._totals.missing), num_samples, self._totals.missing,
                f" (checkpoint: {ckpt_path})" if ckpt_path else "",
            )
        self._totals.completed = len(done)

        # Seed-order merging keeps metrics deterministic across execution orders.
        records: List[ItemRecord] = []
        for idx in done:
            record = ItemRecord(index=idx, extra={"result": results[idx]})
            self.accumulate(record)
            records.append(record)
        return records

    def accumulate(self, record: ItemRecord) -> None:
        r = record.extra["result"]
        self._totals.sample_logs.append(r["sample_log"])
        self._totals.solve_depths.append(r["correct_steps"])
        self._totals.perfect_flags.append(bool(r["perfect_solve"]))
        for i in range(self.n_moves):
            self._totals.per_step_totals[i] += r["per_step_totals"][i]
        for tm, om in r["confusion_pairs"]:
            self._totals.confusion[tm][om] += 1
        self.latencies.extend(r["latencies"])
        if r["first_error_step"] is not None:
            self._totals.first_error_step.append(r["first_error_step"])

    # ----- Aggregation -----

    def aggregate(self, records: List[ItemRecord], num_samples: int) -> Dict[str, Any]:
        totals = self._totals
        solve_depths = totals.solve_depths
        perfect_flags = totals.perfect_flags
        all_sample_logs = totals.sample_logs
        missing, done = totals.missing, range(totals.completed)

        scored = scoring.score(
            self.test_type,
            records,
            context={"steps": self.n_moves, "idk_weight": self.idk_weight},
        )

        avg_depth = (sum(solve_depths) / len(solve_depths)) if solve_depths else 0.0
        perfect = sum(perfect_flags)
        first_err_hist = Counter(self._totals.first_error_step)
        avg_latency = (sum(self.latencies) / len(self.latencies)) if self.latencies else 0.0

        repeat_prev, repeat_any, revisited, t2plus_decisions = self._cycling_stats(
            all_sample_logs)

        return {
            "scoring": "oracle-optimal move from the state actually reached",
            "exact_scramble_depth": True,
            "arm": "history" if self.history_enabled else "markov",
            "history_config": {
                "enabled": self.history_enabled,
                "images": self.history_images,
                "full_responses": self.history_full_responses,
            },
            "generation_config": {
                "thinking_budget": self.thinking_budget,
            },
            "seed_config": {
                "seed_run": self.seed_run,
                "seed_prefix": self.seed_prefix if self._seed_steps else 0,
            },
            "n_moves_scrambled": self.n_moves,
            "average_solve_depth": avg_depth,
            "perfect_solves_ratio": perfect / max(1, len(solve_depths)),
            "step_accuracy": scored["step_accuracy"],
            "first_error_hist": dict(first_err_hist),
            "confusion_matrix": {k: dict(v) for k, v in self._totals.confusion.items()},
            "avg_latency_ms": avg_latency * 1000,
            "num_samples": len(solve_depths),
            "requested_samples": num_samples,
            "completed_samples": len(done),
            "missing_seeds": missing,
            "complete": not missing,
            "selective": scored["selective"],
            "abstention": {
                "enabled": self.idk_enabled,
                "policy": self.idk_policy,
                "idk_weight": self.idk_weight,
                "apa": scored["apa"],
                "conf_threshold": self.idk_conf_threshold,
            },
            "parse_failures": scored["parse_failures"],
            "parse_fail_rate": scored["parse_fail_rate"],
            "step_position_split": scored["step_position_split"],
            "cycling": {
                "repeat_prev_move": repeat_prev,
                "repeat_any_move": repeat_any,
                "revisited_state": revisited,
                "n_t2plus_decisions": t2plus_decisions,
                "repeat_any_move_rate": (repeat_any / t2plus_decisions) if t2plus_decisions else 0.0,
                "revisited_state_rate": (revisited / t2plus_decisions) if t2plus_decisions else 0.0,
            },
            "samples": all_sample_logs,
        }

    def _cycling_stats(self, sample_logs: List[Dict[str, Any]]):
        """Count repeated moves and revisited states from step two onwards."""
        repeat_prev = repeat_any = revisited = 0
        t2plus_decisions = 0
        for s in sample_logs:
            moves: List[str] = []
            states: List[str] = []
            for step in s["steps_data"]:
                mv = step.get("chosen_move")
                if mv is None:
                    continue
                st = step.get("cube_state")
                if moves:
                    t2plus_decisions += 1
                    repeat_prev += int(mv == moves[-1])
                    repeat_any += int(mv in moves)
                    revisited += int(st in states)
                moves.append(mv)
                states.append(st)
        return repeat_prev, repeat_any, revisited, t2plus_decisions

    # ----- Reporting -----

    def summary(self, payload: Dict[str, Any], records: List[ItemRecord]) -> None:
        solve_depths = self._totals.solve_depths
        perfect = sum(self._totals.perfect_flags)
        selective = payload["selective"]
        split = payload["step_position_split"]
        cycling = payload["cycling"]

        self._logger.info(
            "Average oracle-optimal steps (TA): %.2f / %s",
            payload["average_solve_depth"], self.n_moves)
        self._logger.info(f"Perfect Solves: {perfect}/{len(solve_depths)} "
                          f"({(perfect/len(solve_depths))*100:.2f}%)"
                          if solve_depths else "Perfect Solves: 0/0")
        self._logger.info("Per-step accuracy: %s | Per-step Ns: %s",
                          [round(x, 3) for x in payload["step_accuracy"]],
                          [int(t) for t in self._totals.per_step_totals])
        self._logger.info("Avg latency: %.1f ms", payload["avg_latency_ms"])
        self._logger.info(
            "Selective metrics - coverage=%.3f, selective_acc=%.3f, IDK=%d, APA(%.2f)=%.3f",
            selective["coverage_overall"], selective["selective_accuracy"],
            selective["n_idk"], self.idk_weight, payload["abstention"]["apa"],
        )
        self._logger.info(
            "Arm=%s - parse_fail_rate=%.3f | t=1 acc=%.3f (n=%d) | t>=2 acc=%.3f (n=%d) | "
            "repeat_any=%d/%d, revisited_state=%d/%d",
            payload["arm"],
            payload["parse_fail_rate"], split["t1_accuracy"], split["t1_n"],
            split["t2plus_accuracy"], split["t2plus_n"],
            cycling["repeat_any_move"], cycling["n_t2plus_decisions"],
            cycling["revisited_state"], cycling["n_t2plus_decisions"],
        )

    # ----- Results -----

    def result_filename(self) -> Optional[str]:
        return "step_by_step_history.json" if self.history_enabled else None
