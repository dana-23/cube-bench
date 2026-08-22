from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from ..core import BaseTest
from cube_bench.prompts.prompt_factory import PromptFactory
from cube_bench.sim.cube_simulator import VirtualCube

logger = logging.getLogger(__name__)


class StepByStepTest(BaseTest):
    """Closed-loop: ask for next move at each step, accept optimal or teacher."""

    test_type = "step_by_step"

    def __init__(
        self,
        assistant,
        config,
        n_moves: int,
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
        self.per_step_totals: List[int] = [0] * n_moves
        self.per_step_correct: List[int] = [0] * n_moves
        self.first_error_step: List[int] = []
        self.confusion = defaultdict(Counter)
        self.parse_failures = 0

        self.idk_enabled = bool(idk_enabled)
        self.idk_weight = float(idk_weight)
        self.idk_policy = str(idk_policy)
        self.idk_conf_threshold = 50  # kept from prior behavior
        self.per_step_idk: List[int] = [0] * n_moves

        # Arm B (history-conditioned): at step t the model sees the whole episode
        # so far as a multi-turn exchange — S0, a0, S1, a1, …, St. Arm A (Markov,
        # the default) shows only the current state.
        self.history_enabled = bool(history_enabled)
        self.history_images = bool(history_images)
        self.history_full_responses = bool(history_full_responses)

        # Episodes are independent (each keyed by its own seed=idx), so they can
        # run concurrently; steps WITHIN an episode stay serial (closed-loop).
        # >1 is for the remote API models (Claude/Gemini/OpenAI), whose generate()
        # builds a fresh client per call and is safe to call from many threads.
        # Keep at 1 for local HF/vLLM models.
        self.concurrency = max(1, int(concurrency))

        # Per-episode checkpointing: each completed episode is appended to a JSONL
        # sidecar as soon as it finishes, so an interruption (crash, Ctrl-C, spend
        # limit) never loses completed work. Re-running the same config resumes,
        # skipping episodes already on disk. Keyed by model/arm/n_moves/samples.
        self.checkpoint = bool(checkpoint)
        self.checkpoint_dir = checkpoint_dir

        # Seed/replay: reuse the first ``seed_prefix`` steps of each episode from a
        # prior run (``seed_run`` JSON), then continue the closed loop in THIS arm's
        # mode. Only valid where the seed's decision was made under conditions
        # identical to this arm — i.e. the pre-history prefix. For a Markov arm
        # seeded from a History run that means seed_prefix=1 (t=1 has no history),
        # which guarantees the two arms share an identical t=1 and enter t=2 from
        # the same state — a clean, item-paired comparison at the first step where
        # history can act.
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
            logger.info(f"IDK Enabled -- Policy: {self.idk_policy} -- Weight: {self.idk_weight}")
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
            logger.info(f"Model's full response:\n\n{text}")
            return False, None
        return (pred == gold_letter), pred

    def _build_prompts(self, kwargs: Dict[str, Any]) -> Tuple[str, str]:
        """(sys, user). If IDK is enabled, use the abstention-aware template; otherwise PromptFactory."""
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
            "1) For each candidate, internally simulate that move on the textual state and estimate the resulting distance d1 (HTM).\n"
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

    def _run_episode(self, idx: int) -> Dict[str, Any]:
        """Run one full episode (seed=idx). Steps stay serial; all counters are
        episode-local and returned for the caller to merge, so the method mutates
        no shared state and is safe to call from many threads at once."""
        cube = VirtualCube()
        scramble = cube.scramble(random_seed=idx, n_moves=self.n_moves)
        solution_path = str(deepcopy(scramble).reverse()).split()
        correct_steps = 0
        teacher_help = 0

        if self.verbose:
            logger.info(f"[sample {idx}] Scramble: {scramble}")
            logger.info(f"[sample {idx}] Teacher path: {solution_path}")

        sample_log = {
            "sample_id": idx,
            "scramble": str(scramble),
            "solution_path": solution_path,
            "steps_data": [],
        }

        # Episode transcript for the history arm: alternating user/assistant
        # turns, one pair per prior step.
        convo: List[Dict[str, Any]] = []

        # Episode-local accumulators (merged into shared state after the episode).
        per_step_totals = [0] * self.n_moves
        per_step_correct = [0] * self.n_moves
        per_step_idk = [0] * self.n_moves
        confusion_pairs: List[Tuple[str, str]] = []
        latencies: List[float] = []
        n_correct = n_wrong = n_idk = 0
        total_decisions = 0
        parse_failures = 0
        first_error_step: Optional[int] = None

        for step_i, teacher_move in enumerate(solution_path):
            if cube.is_solved():
                break

            state_text = self.state_text(cube)
            state_img = cube.to_image()

            seed_bytes = f"{self.n_moves}:{idx}:{step_i}".encode()
            seed_int = int.from_bytes(hashlib.sha256(seed_bytes).digest()[:8], "big")
            step_rng = random.Random(seed_int)

            options, gold_letter = self.gen_mcq_balanced(cube, teacher_move, step_rng)

            kwargs = {
                "n_moves": self.n_moves - correct_steps - teacher_help,
                "n_moves_optimal": (self.n_moves - 1) - correct_steps - teacher_help,
                "textual_representation": state_text,
                "move_A": options["A"],
                "move_B": options["B"],
                "move_C": options["C"],
                "move_D": options["D"],
                "metric": "HTM (Half-Turn Metric)",
            }
            sys_prompt, user_prompt = self._build_prompts(kwargs)

            seed_steps = self._seed_steps.get(idx)
            replay = (
                seed_steps is not None
                and step_i < self.seed_prefix
                and step_i < len(seed_steps)
            )

            if replay:
                # Reuse the prior run's decision at this step. It is only valid to
                # do so where conditions were identical to this arm; we assert the
                # regenerated state and MCQ match the seed's to guarantee that.
                seed = seed_steps[step_i]
                if state_text != seed["cube_state"]:
                    raise AssertionError(f"[ep {idx} step {step_i}] seed state mismatch (scramble/seeds drifted)")
                if options != seed["options"] or gold_letter != seed["correct_letter"]:
                    raise AssertionError(f"[ep {idx} step {step_i}] seed MCQ mismatch (MCQ generation drifted)")
                resp = seed.get("full_response", "")
                pred_letter = seed.get("predicted_letter")
                is_correct = bool(seed.get("is_correct"))
            else:
                history = None
                if self.history_enabled and convo:
                    if self.history_images:
                        history = list(convo)
                    else:
                        # Text-serialized history: prior states as text only;
                        # the current state keeps its image.
                        history = [{k: v for k, v in t.items() if k != "image"} for t in convo]

                t0 = time.time()
                resp = self.ask(
                    user_prompt=user_prompt,
                    system_prompt=sys_prompt,
                    image=state_img,
                    history=history,
                    track_latency=False,
                )
                latencies.append(time.time() - t0)

                is_correct, pred_letter = self._eval(resp, gold_letter)

            options_move = options.get(pred_letter) if pred_letter and pred_letter != "IDK" else None

            total_decisions += 1
            per_step_totals[step_i] += 1

            if self.history_enabled:
                convo.append({"role": "user", "text": user_prompt, "image": state_img})
                if self.history_full_responses or not pred_letter or pred_letter == "IDK":
                    assistant_text = resp
                else:
                    assistant_text = f"<ANSWER> {pred_letter} </ANSWER>"
                convo.append({"role": "assistant", "text": assistant_text})

            if pred_letter is None:
                parse_failures += 1
                sample_log["steps_data"].append({
                    "step": step_i,
                    "cube_state": state_text,
                    "options": options,
                    "correct_letter": gold_letter,
                    "full_response": resp,
                    "predicted_letter": None,
                    "is_correct": False,
                    "parse_fail": True,
                })
                first_error_step = step_i + 1
                if self.verbose:
                    logger.info(f"[sample {idx}] Parse failure at step {step_i + 1}; ending episode.")
                break

            if self.idk_enabled and pred_letter == "IDK":
                logger.info("Model responded with IDK.")
                n_idk += 1
                per_step_idk[step_i] += 1

                sample_log["steps_data"].append({
                    "step": step_i,
                    "cube_state": state_text,
                    "options": options,
                    "correct_letter": gold_letter,
                    "full_response": resp,
                    "predicted_letter": "IDK",
                    "is_correct": False,
                    "abstained": True,
                    "idk_policy": self.idk_policy,
                })

                if self.idk_policy == "teacher_on_abstain":
                    cube.apply(teacher_move)
                    teacher_help += 1
                    continue
                else:
                    first_error_step = step_i + 1
                    break

            if self.verbose:
                logger.info(f"Model's chosen option: {pred_letter} -> {options_move}")

            per_step_correct[step_i] += int(is_correct)

            if pred_letter and pred_letter != "IDK":
                if is_correct:
                    n_correct += 1
                else:
                    n_wrong += 1

            if options_move is not None:
                confusion_pairs.append((teacher_move, options_move))

            sample_log["steps_data"].append({
                "step": step_i,
                "cube_state": state_text,
                "options": options,
                "correct_letter": gold_letter,
                "full_response": resp,
                "predicted_letter": pred_letter,
                "chosen_move": options_move,
                "is_correct": is_correct,
            })

            good_moves = self.optimal_first_moves(cube)
            if self.verbose:
                logger.info(f"[Sample: {idx} Step: {step_i}] Oracle-good moves: {sorted(good_moves)}")

            made_progress = False
            if options_move is not None:
                made_progress, _, _ = self.move_makes_progress(cube, options_move)

            if options_move and (is_correct or options_move in good_moves):
                cube.apply(options_move)
                if is_correct:
                    correct_steps += 1
            else:
                if options_move and made_progress:
                    cube.apply(options_move)
                else:
                    first_error_step = step_i + 1
                    if self.verbose:
                        logger.info(f"[sample {idx}] First error at step {step_i + 1}")
                    break

            # Validate trajectory reconstruction: after applying a replayed move,
            # the rebuilt state must equal the seed run's logged next state.
            if replay and step_i + 1 < len(seed_steps):
                rebuilt = self.state_text(cube)
                expected = seed_steps[step_i + 1]["cube_state"]
                if rebuilt != expected:
                    raise AssertionError(
                        f"[ep {idx}] rebuilt S{step_i + 1} != seed's logged state "
                        "(replayed move application diverged)"
                    )

        return {
            "sample_log": sample_log,
            "correct_steps": correct_steps,
            "per_step_totals": per_step_totals,
            "per_step_correct": per_step_correct,
            "per_step_idk": per_step_idk,
            "confusion_pairs": confusion_pairs,
            "latencies": latencies,
            "n_correct": n_correct,
            "n_wrong": n_wrong,
            "n_idk": n_idk,
            "total_decisions": total_decisions,
            "parse_failures": parse_failures,
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
        """Stable, config-keyed JSONL path (independent of the per-run timestamped
        output dir), so re-runs of the same config find the prior progress."""
        if not self.checkpoint:
            return None
        base = Path(self.checkpoint_dir) if self.checkpoint_dir else Path.cwd() / "checkpoints" / "step_by_step"
        arm = "history" if self.history_enabled else "markov"
        # A seeded/branched run is a distinct experiment from a plain one — key it
        # separately so it never resumes from (or overwrites) a plain-run checkpoint.
        seed_tag = f"_seeded-{Path(self.seed_run).stem}-p{self.seed_prefix}" if self._seed_steps else ""
        key = f"{self.assistant.get_name()}_{arm}_d{self.n_moves}_n{num_samples}{seed_tag}"
        safe = "".join(c if (c.isalnum() or c in "-._") else "_" for c in key)
        base.mkdir(parents=True, exist_ok=True)
        return base / f"{safe}.jsonl"

    @staticmethod
    def _load_checkpoint(path: Path) -> Dict[int, Dict[str, Any]]:
        results: Dict[int, Dict[str, Any]] = {}
        if not path.exists():
            return results
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                results[int(rec["sample_id"])] = rec["result"]
            except Exception as e:  # tolerate a torn last line from a hard kill
                logger.warning("Skipping malformed checkpoint line: %s", e)
        return results

    def run(self, num_samples: int):
        desc = f"Step-by-step ({self.n_moves} moves)"

        # Resume: load episodes already completed by a prior run of this config.
        ckpt_path = self._checkpoint_path(num_samples)
        results: Dict[int, Dict[str, Any]] = {}
        if ckpt_path is not None:
            results = self._load_checkpoint(ckpt_path)
            if results:
                logger.info("Resuming from %s — %d/%d episodes already done",
                            ckpt_path, len(results), num_samples)

        todo = [idx for idx in range(num_samples) if idx not in results]
        ckpt_lock = threading.Lock()

        def _record(idx: int, r: Dict[str, Any]) -> None:
            results[idx] = r
            if ckpt_path is not None:
                with ckpt_lock, open(ckpt_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"sample_id": idx, "result": r}) + "\n")

        # Episodes are independent, so run them concurrently when asked; steps
        # within each episode stay serial (the loop is closed-loop / path-dependent).
        # A permanently-failed episode (e.g. retries exhausted) is logged and left
        # out; it stays absent from the checkpoint so a later re-run retries just it.
        if todo:
            if self.concurrency > 1:
                logger.info("Running %d episodes with concurrency=%d", len(todo), self.concurrency)
                with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
                    futures = {pool.submit(self._run_episode, idx): idx for idx in todo}
                    for fut in tqdm(as_completed(futures), total=len(todo), desc=desc):
                        idx = futures[fut]
                        try:
                            _record(idx, fut.result())
                        except Exception as e:
                            logger.error("Episode %d failed permanently: %s", idx, e)
            else:
                for idx in tqdm(todo, desc=desc):
                    try:
                        _record(idx, self._run_episode(idx))
                    except Exception as e:
                        logger.error("Episode %d failed permanently: %s", idx, e)

        done = sorted(results)
        missing = [idx for idx in range(num_samples) if idx not in results]
        if missing:
            logger.warning(
                "%d/%d episodes incomplete: %s — re-run the same command to resume%s.",
                len(missing), num_samples, missing,
                f" (checkpoint: {ckpt_path})" if ckpt_path else "",
            )

        # Merge episode results into shared state in seed order — inputs are
        # seeded by idx, so the saved metrics are identical regardless of the
        # execution order (only wall-clock latency values differ).
        solve_depths: List[int] = []
        all_sample_logs: List[Dict[str, Any]] = []
        n_correct = n_wrong = n_idk = 0
        total_decisions = 0

        for idx in done:
            r = results[idx]
            all_sample_logs.append(r["sample_log"])
            solve_depths.append(r["correct_steps"])
            for i in range(self.n_moves):
                self.per_step_totals[i] += r["per_step_totals"][i]
                self.per_step_correct[i] += r["per_step_correct"][i]
                self.per_step_idk[i] += r["per_step_idk"][i]
            for tm, om in r["confusion_pairs"]:
                self.confusion[tm][om] += 1
            self.latencies.extend(r["latencies"])
            if r["first_error_step"] is not None:
                self.first_error_step.append(r["first_error_step"])
            self.parse_failures += r["parse_failures"]
            n_correct += r["n_correct"]
            n_wrong += r["n_wrong"]
            n_idk += r["n_idk"]
            total_decisions += r["total_decisions"]

        # -------- Aggregate --------
        avg_depth = (sum(solve_depths) / len(solve_depths)) if solve_depths else 0.0
        perfect = sum(1 for d in solve_depths if d == self.n_moves)
        step_acc = [c / t if t else 0.0 for c, t in zip(self.per_step_correct, self.per_step_totals)]
        first_err_hist = Counter(self.first_error_step)
        avg_latency = (sum(self.latencies) / len(self.latencies)) if self.latencies else 0.0

        answered = n_correct + n_wrong
        coverage_overall = (answered / total_decisions) if total_decisions else 0.0
        selective_acc = (n_correct / answered) if answered else 0.0
        apa = ((n_correct + self.idk_weight * n_idk) / total_decisions) if total_decisions else 0.0
        coverage_by_step = [((t - z) / t) if t else 0.0 for t, z in zip(self.per_step_totals, self.per_step_idk)]

        parse_fail_rate = (self.parse_failures / total_decisions) if total_decisions else 0.0

        # Step-position split: at t=1 the arms are identical by construction
        # (no history exists yet), so any history effect can only show at t>=2.
        t1_total = self.per_step_totals[0] if self.per_step_totals else 0
        t1_acc = (self.per_step_correct[0] / t1_total) if t1_total else 0.0
        t2_correct = sum(self.per_step_correct[1:])
        t2_total = sum(self.per_step_totals[1:])
        t2_acc = (t2_correct / t2_total) if t2_total else 0.0

        # Cycling: does the model re-pick moves / revisit states within an episode?
        repeat_prev = repeat_any = revisited = 0
        t2plus_decisions = 0
        for s in all_sample_logs:
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

        logger.info(f"Average Correct Steps (teacher-adherence): {avg_depth:.2f} / {self.n_moves}")
        logger.info(f"Perfect Solves: {perfect}/{len(solve_depths)} "
                    f"({(perfect/len(solve_depths))*100:.2f}%)" if solve_depths else "Perfect Solves: 0/0")
        logger.info("Per-step accuracy: %s | Per-step Ns: %s",
                    [round(x, 3) for x in step_acc], [int(t) for t in self.per_step_totals])
        logger.info(f"Avg latency: {avg_latency*1000:.1f} ms")
        logger.info(
            "Selective metrics — coverage=%.3f, selective_acc=%.3f, IDK=%d, APA(%.2f)=%.3f",
            coverage_overall, selective_acc, n_idk, self.idk_weight, apa,
        )
        logger.info(
            "Arm=%s — parse_fail_rate=%.3f | t=1 acc=%.3f (n=%d) | t>=2 acc=%.3f (n=%d) | "
            "repeat_any=%d/%d, revisited_state=%d/%d",
            "history" if self.history_enabled else "markov",
            parse_fail_rate, t1_acc, t1_total, t2_acc, t2_total,
            repeat_any, t2plus_decisions, revisited, t2plus_decisions,
        )

        self.save({
            "arm": "history" if self.history_enabled else "markov",
            "history_config": {
                "enabled": self.history_enabled,
                "images": self.history_images,
                "full_responses": self.history_full_responses,
            },
            "seed_config": {
                "seed_run": self.seed_run,
                "seed_prefix": self.seed_prefix if self._seed_steps else 0,
            },
            "n_moves_scrambled": self.n_moves,
            "average_solve_depth": avg_depth,
            "perfect_solves_ratio": perfect / max(1, len(solve_depths)),
            "step_accuracy": step_acc,
            "first_error_hist": dict(first_err_hist),
            "confusion_matrix": {k: dict(v) for k, v in self.confusion.items()},
            "avg_latency_ms": avg_latency * 1000,
            "num_samples": len(solve_depths),
            "requested_samples": num_samples,
            "completed_samples": len(done),
            "missing_seeds": missing,
            "complete": not missing,
            "selective": {
                "coverage_overall": coverage_overall,
                "coverage_by_step": coverage_by_step,
                "selective_accuracy": selective_acc,
                "n_correct": n_correct,
                "n_wrong": n_wrong,
                "n_idk": n_idk,
                "total_decisions": total_decisions,
            },
            "abstention": {
                "enabled": self.idk_enabled,
                "policy": self.idk_policy,
                "idk_weight": self.idk_weight,
                "apa": apa,
                "conf_threshold": self.idk_conf_threshold,
            },
            "parse_failures": self.parse_failures,
            "parse_fail_rate": parse_fail_rate,
            "step_position_split": {
                "t1_accuracy": t1_acc,
                "t1_n": t1_total,
                "t2plus_accuracy": t2_acc,
                "t2plus_n": t2_total,
            },
            "cycling": {
                "repeat_prev_move": repeat_prev,
                "repeat_any_move": repeat_any,
                "revisited_state": revisited,
                "n_t2plus_decisions": t2plus_decisions,
                "repeat_any_move_rate": (repeat_any / t2plus_decisions) if t2plus_decisions else 0.0,
                "revisited_state_rate": (revisited / t2plus_decisions) if t2plus_decisions else 0.0,
            },
            "samples": all_sample_logs,
        }, filename="step_by_step_history.json" if self.history_enabled else None)
