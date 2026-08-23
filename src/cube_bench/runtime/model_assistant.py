"""Modular model-strategy framework.

- One registry line -> new model
- Shared prompt builder & utilities
- HuggingFace (HF) or vLLM engines
- Vision-ready (PIL or Path), efficient, and multi-GPU friendly
"""

from __future__ import annotations

import logging
import os
import random
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Type, Union

import torch
import yaml
from dotenv import load_dotenv
from PIL import Image

from cube_bench.sim.cube_simulator import VirtualCube

torch.set_float32_matmul_precision("high")
load_dotenv()

logger = logging.getLogger("assistant")
logging.basicConfig(
    level=os.getenv("LOGLEVEL", "INFO"),
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)

# ------------------------------------------------------------------------------
# Transient-error retry (for the remote API strategies under concurrency)
# ------------------------------------------------------------------------------
# Tunable via env; defaults give ~1+5 tries with exponential backoff + jitter.
_RETRY_MAX = int(os.getenv("CUBE_BENCH_API_MAX_RETRIES", "5"))
_RETRY_BASE_SEC = float(os.getenv("CUBE_BENCH_API_RETRY_BASE_SEC", "2.0"))
_RETRY_CAP_SEC = float(os.getenv("CUBE_BENCH_API_RETRY_CAP_SEC", "60.0"))

# HTTP statuses worth retrying (rate-limit / timeout / transient server errors;
# 529 = Anthropic "overloaded").
_TRANSIENT_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}
# Substrings matched against the exception type name / code / message. Covers the
# Anthropic, OpenAI and google-genai SDKs without importing any of them here.
_TRANSIENT_TOKENS = (
    "ratelimit", "overloaded", "timeout", "timedout", "connection",
    "internalserver", "serviceunavailable", "servererror", "apierror",
    "deadlineexceeded", "resourceexhausted", "unavailable", "temporarily",
)


def _is_transient(exc: Exception) -> bool:
    """Heuristic: is this exception a retryable transient API failure?"""
    codes = []
    for attr in ("status_code", "code", "http_status", "status"):
        v = getattr(exc, attr, None)
        if isinstance(v, bool):
            continue
        if isinstance(v, int):
            codes.append(v)
        elif isinstance(v, str) and v.strip().isdigit():
            codes.append(int(v.strip()))
    if any(c in _TRANSIENT_STATUS for c in codes):
        return True
    hay = f"{type(exc).__name__} {getattr(exc, 'code', '')} {exc}".lower()
    return any(tok in hay for tok in _TRANSIENT_TOKENS)


class _RateLimiter:
    """Thread-safe rolling-window rate limiter: at most ``rpm`` acquisitions per
    any 60s window, shared across all worker threads. rpm <= 0 disables it."""

    def __init__(self, rpm: int):
        self.rpm = int(rpm)
        self._lock = threading.Lock()
        self._times: deque = deque()

    def acquire(self) -> None:
        """Block until another request fits inside the trailing 60-second window."""
        if self.rpm <= 0:
            return
        while True:
            with self._lock:
                now = time.time()
                while self._times and now - self._times[0] >= 60.0:
                    self._times.popleft()
                if len(self._times) < self.rpm:
                    self._times.append(now)
                    return
                sleep_for = 60.0 - (now - self._times[0])
            time.sleep(max(0.0, sleep_for) + 0.001)  # sleep OUTSIDE the lock


# Client-side request cap (requests/min) to stay under a provider RPM limit.
# 0 = unlimited (default). Set e.g. CUBE_BENCH_API_RPM=140 for a 150 RPM plan.
_RPM_LIMIT = int(os.getenv("CUBE_BENCH_API_RPM", "0"))
_RATE_LIMITER = _RateLimiter(_RPM_LIMIT)


def _call_with_retry(fn, label: str):
    """Call ``fn`` (a no-arg thunk); retry transient failures with exponential
    backoff + jitter. Non-transient errors and the final failure re-raise.
    Each attempt (including retries) passes through the global rate limiter."""
    delay = _RETRY_BASE_SEC
    for attempt in range(1, _RETRY_MAX + 2):  # 1 initial try + _RETRY_MAX retries
        try:
            _RATE_LIMITER.acquire()
            return fn()
        except Exception as exc:  # noqa: BLE001 — classify then re-raise if not transient
            if attempt > _RETRY_MAX or not _is_transient(exc):
                raise
            sleep_s = min(_RETRY_CAP_SEC, delay) * (0.5 + random.random())  # 0.5x–1.5x jitter
            logger.warning(
                "[%s] transient API error (try %d/%d): %s: %s — retrying in %.1fs",
                label, attempt, _RETRY_MAX + 1, type(exc).__name__, exc, sleep_s,
            )
            time.sleep(sleep_s)
            delay = min(_RETRY_CAP_SEC, delay * 2)
    raise AssertionError(f"[{label}] retry loop exited without returning or raising")


# Optional imports (keep file importable without deps)
try:
    from transformers import AutoProcessor
except Exception as e:  # pragma: no cover
    AutoProcessor = None  # type: ignore
    logger.info("[transformers] Import optional: %s", e)

try:
    from vllm import LLM, SamplingParams
except Exception as e:  # pragma: no cover
    LLM = SamplingParams = None  # type: ignore
    logger.info("[vLLM] Import: %s", e)

# Keep vLLM quiet but preserve our INFO logs
logging.getLogger("vllm").setLevel(logging.WARNING)
logging.getLogger("vllm.core").setLevel(logging.WARNING)


# 1) Dataclasses & small utilities

@dataclass(frozen=True)
class ModelSpec:
    """One registry entry: where a model lives and which strategy runs it."""

    name: str
    path: str                 # local path, HF repo, or API name
    strategy_hf: Optional[Type["ModelStrategy"]] = None
    strategy_vllm: Optional[Type["ModelStrategy"]] = None
    dtype: torch.dtype = torch.bfloat16
    supports_image: bool = True

@dataclass
class GenerationConfig:
    """Decoding parameters shared by every strategy."""

    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    do_sample: bool = True

def _as_pil(img: Optional[Union[Image.Image, Path, str]]) -> Optional[Image.Image]:
    """Normalize a single image input to a RGB PIL.Image (or None)."""
    if img is None:
        return None
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    # Accept Path or str
    return Image.open(str(img)).convert("RGB")


def _to_device(batch: Any, device: torch.device, dtype: Optional[torch.dtype] = None) -> Any:
    # Supports plain dicts or HF BatchFeature (which has .to)
    try:
        return batch.to(device)
    except Exception:
        pass
    if isinstance(batch, dict):
        out = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                if dtype and v.dtype.is_floating_point:
                    out[k] = v.to(device, dtype=dtype, non_blocking=True)
                else:
                    out[k] = v.to(device, non_blocking=True)
            else:
                out[k] = v
        return out
    return batch


# 2) Prompt builder (single source of truth)

class PromptBuilder:
    """Turns (system, user, image, reference) into the tensors a HF model expects."""

    def __init__(self, processor: "AutoProcessor") -> None:
        self.processor = processor

    def build_hf_inputs(
        self,
        user_prompt: str,
        system_prompt: str,
        image: Optional[Union[Image.Image, Path]] = None,
        reference: str = "",
    ) -> Dict[str, Any]:
        """Build the tokenized, device-ready inputs for a HuggingFace model."""
        pil = _as_pil(image)

        # Compose multimodal-style messages
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": ([{"type": "image", "image": pil}] if pil is not None else []) +
                                    [{"type": "text", "text": user_prompt}]},
        ]
        if reference:
            messages.append({"role": "assistant", "content": [{"type": "text", "text": reference}]})

        # Prefer true multimodal chat templates (works if you loaded AutoProcessor for a *-hf model)
        try:
            return self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                return_dict=True,
            )
        except TypeError as e:
            # Template is text-only → flatten “parts” and process images separately
            if "concatenate str (not \"list\") to str" not in str(e):
                raise

            tok = getattr(self.processor, "tokenizer", self.processor)  # tokenizer for text
            image_processor = getattr(self, "image_processor", getattr(self.processor, "image_processor", None))

            # Flatten content -> string + image placeholders
            image_token = getattr(tok, "image_token", "<image>")
            flat_msgs, images = [], []
            for m in messages:
                parts = []
                for p in m.get("content", []):
                    t = p.get("type")
                    if t == "image":
                        parts.append(image_token)
                        images.append(p.get("image") or p.get("url") or p.get("path"))
                    elif t == "text":
                        parts.append(p.get("text", ""))
                flat_msgs.append({"role": m["role"], "content": "".join(parts)})

            chat_text = tok.apply_chat_template(flat_msgs, add_generation_prompt=True, tokenize=False)

            # ALWAYS tokenize text with the tokenizer (no images kwarg here!)
            text_inputs = tok(chat_text, return_tensors="pt", padding=True)

            # If there’s an image, process it with an image processor and merge
            if images:
                if image_processor is None:
                    raise RuntimeError(
                        "Your self.processor is a tokenizer (no images=). "
                        "Load a multimodal AutoProcessor for the model or attach self.image_processor."
                    ) from e
                vision_inputs = image_processor(images=images, return_tensors="pt")
                text_inputs.update(vision_inputs)

            return text_inputs


    def build_vllm_request(self, user_prompt: str, system_prompt: str,
                       image: Optional[Union[Image.Image, Path]] = None) -> Dict[str, Any]:
        """Build the prompt dict a vLLM engine expects, with image placeholders."""
        pil = _as_pil(image)
        # Use HF multimodal chat template to insert the placeholder token(s)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",
            "content": ([{"type": "image"}] if pil is not None else []) + [{"type": "text", "text": user_prompt}]},
        ]
        prompt_txt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        req = {"prompt": prompt_txt}
        if pil is not None:
            req["multi_modal_data"] = {"image": pil}  # PIL image directly (no temp file)
        return req



# 3) Strategy base class

class ModelStrategy(ABC):
    """Backend contract: load a model, generate from it, then release it."""

    def __init__(self, spec: ModelSpec):
        self.spec = spec
        self.processor: Optional["AutoProcessor"] = None
        self.model: Any = None     # HF model OR vLLM engine
        self.prompt_builder: Optional[PromptBuilder] = None
        self._device_for_inputs: str = "cpu"   # where to stage inputs

    #  Lifecycle hooks
    @abstractmethod
    def load(self) -> None:
        """Load the model. No-op for strategies that connect lazily."""

    @abstractmethod
    def generate(
        self,
        user_prompt: str,
        system_prompt: str,
        image: Optional[Union[Image.Image, Path]],
        gen_cfg: GenerationConfig,
        reference: str = "",
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """``history``: ordered prior turns prepended before the final
        (user_prompt, image) turn. Each turn is
        ``{"role": "user"|"assistant", "text": str, "image": Optional[PIL|Path]}``
        (``image`` only on user turns). Only the API strategies support it."""

    #  Helpers
    def _ensure_processor(self) -> None:
        assert AutoProcessor is not None, "transformers not installed"
        # InternVL needs its remote (non-fast) tokenizer with image tokens.
        try:
            self.processor = AutoProcessor.from_pretrained(
                self.spec.path,
                trust_remote_code=True,
                use_fast=True
            )

        except Exception as e:
            logger.warning("Could not load fast tokenizer for %s: %s", self.spec.name, e)
            logger.warning("Falling back to slow tokenizer.")

            self.processor = AutoProcessor.from_pretrained(
                self.spec.path,
                trust_remote_code=True,
                # use_fast=False
            )

        self.prompt_builder = PromptBuilder(self.processor)

    def cleanup(self) -> None:
        """Drop references to the model, processor and prompt builder."""
        logger.debug("[%s] cleanup", self.spec.name)
        try:
            del self.model, self.processor, self.prompt_builder
        except Exception:
            pass
        self.model = self.processor = self.prompt_builder = None  # type: ignore
        torch.cuda.empty_cache()


# 4) HuggingFace base strategy (shared generation & optional batching)

class HuggingFaceStrategy(ModelStrategy):
    """Runs a model in-process through Transformers."""

    def load(self) -> None:
        self._ensure_processor()
        self.model = self._load_model_instance()

        # If model is sharded we won't have single .device; inputs go to cuda:0 if available
        if torch.cuda.is_available():
            self._device_for_inputs = "cuda:0"
        elif torch.backends.mps.is_available():
            self._device_for_inputs = "mps"
        else:
            self._device_for_inputs = "cpu"

        logger.info("[%s] HF model ready (inputs on %s)", self.spec.name, self._device_for_inputs)

    @abstractmethod
    def _load_model_instance(self):
        "To be implemented in model class individually."

    def generate(
        self,
        user_prompt: str,
        system_prompt: str,
        image: Optional[Union[Image.Image, Path]],
        gen_cfg: GenerationConfig,
        reference: str = "",
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:

        if history:
            raise NotImplementedError(
                f"[{self.spec.name}] history-conditioned generation is only "
                "implemented for the API strategies (Claude/Gemini/OpenAI)."
            )
        assert self.prompt_builder is not None and self.processor is not None
        inputs = self.prompt_builder.build_hf_inputs(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            image=image if self.spec.supports_image else None,
            reference=reference,
        )

        # 1) Move to the right device (do NOT cast yet; HF BatchFeature.to() ignores dtype anyway)
        inputs = _to_device(inputs, torch.device(self._device_for_inputs), dtype=None)

        # 2) Align vision input dtype with the model’s vision tower (prevents float vs bf16 conv2d crash)
        vision_dtype = None
        vt = getattr(self.model, "vision_tower", None)
        try:
            vision_dtype = next(vt.parameters()).dtype if vt is not None else next(self.model.parameters()).dtype
        except StopIteration:
            vision_dtype = next(self.model.parameters()).dtype
        if "pixel_values" in inputs and torch.is_tensor(inputs["pixel_values"]):
            pv = inputs["pixel_values"]
            if pv.dtype != vision_dtype:
                inputs["pixel_values"] = pv.to(dtype=vision_dtype, non_blocking=True)

        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                do_sample=gen_cfg.do_sample,
                top_p=gen_cfg.top_p,
                max_new_tokens=gen_cfg.max_new_tokens,
                temperature = gen_cfg.temperature
            )

        # Decode only the generated continuation
        gen_ids = out[:, input_len:]
        text = self.processor.decode(gen_ids[0], skip_special_tokens=True)
        return text

    # simple batch API (strings, same system prompt & no images for now)
    def generate_batch(
        self,
        prompts: Iterable[str],
        system_prompt: str = "You are a helpful assistant.",
        gen_cfg: Optional[GenerationConfig] = None,
    ) -> List[str]:
        """Generate one completion per prompt, sharing the system prompt (text only)."""
        assert self.prompt_builder is not None and self.processor is not None
        gen_cfg = gen_cfg or GenerationConfig()
        msgs = [
            [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {"role": "user", "content": [{"type": "text", "text": p}]},
            ]
            for p in prompts
        ]
        # pack via processor
        enc = self.processor.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True, return_tensors="pt", padding=True, return_dict=True
        )
        enc = _to_device(enc, torch.device(self._device_for_inputs), dtype=self.spec.dtype)
        input_lens = (enc["input_ids"] != self.processor.tokenizer.pad_token_id).sum(-1)

        with torch.inference_mode():
            out = self.model.generate(
                **enc,
                do_sample=gen_cfg.do_sample,
                temperature=gen_cfg.temperature,
                top_p=gen_cfg.top_p,
                max_new_tokens=gen_cfg.max_new_tokens,
            )

        results: List[str] = []
        for i in range(out.shape[0]):
            gen_ids = out[i, input_lens[i] :]
            results.append(self.processor.decode(gen_ids, skip_special_tokens=True))
        return results

# concrete HF models

class GemmaStrategy(HuggingFaceStrategy):
    """Gemma-3 loader (``Gemma3ForConditionalGeneration``)."""

    def _load_model_instance(self):
        from transformers import Gemma3ForConditionalGeneration  # type: ignore
        return Gemma3ForConditionalGeneration.from_pretrained(
            self.spec.path, dtype=self.spec.dtype, device_map="auto"
        ).eval()

class LlamaStrategy(HuggingFaceStrategy):
    """Llama-4 loader (``Llama4ForConditionalGeneration``)."""

    def _load_model_instance(self):
        from transformers import Llama4ForConditionalGeneration  # type: ignore
        return Llama4ForConditionalGeneration.from_pretrained(
            self.spec.path, dtype=self.spec.dtype, device_map="auto"
        ).eval()

class QwenVLStrategy(HuggingFaceStrategy):
    """Qwen-VL loader; picks the Qwen3-VL MoE class for the thinking variant."""

    def _load_model_instance(self):
        from transformers import Qwen2_5_VLForConditionalGeneration, Qwen3VLMoeForConditionalGeneration # type: ignore
        if self.spec.name == "qwen3-vl-thinking":
            return Qwen3VLMoeForConditionalGeneration.from_pretrained(
                self.spec.path, dtype="auto", device_map="auto"
            ).eval()
        return Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.spec.path, dtype="auto", device_map="auto"
        ).eval()

    def generate(
        self,
        user_prompt: str,
        system_prompt: str,
        image: Optional[Union[Image.Image, Path]],
        gen_cfg: GenerationConfig,
        reference: str = "",
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        if history:
            raise NotImplementedError(
                f"[{self.spec.name}] history-conditioned generation is only "
                "implemented for the API strategies (Claude/Gemini/OpenAI)."
            )
        model = self.model
        processor = AutoProcessor.from_pretrained(self.spec.path)

        # Build messages like the card example (user role only)
        content = []
        if image is not None:
            if isinstance(image, (str, Path)):
                content.append({"type": "image", "image": str(image)})
            else:  # PIL.Image.Image
                # ensure RGB just in case
                pil = image if image.mode == "RGB" else image.convert("RGB")
                content.append({"type": "image", "image": pil})
        content.append({"type": "text", "text": user_prompt})

        messages = [{"role": "user", "content": content}]

        # Preparation for inference (same args as the card)
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        device = getattr(self.model, "device", torch.device("cuda"))
        inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}

        # Inference: Generation of the output (only max_new_tokens like the card)
        max_new = int(getattr(gen_cfg, "max_new_tokens", 128))
        generated_ids = model.generate(**inputs, max_new_tokens=max_new)

        # Trim prompt tokens from the output (same as the card)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        # Decode (same flags as the card)
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return output_text[0] if output_text else ""


# InternVL3 (HF)

class InternVL3_5Strategy(HuggingFaceStrategy):
    """
    InternVL3-78B-Instruct loader & generator.
    Uses custom device_map splitting for multi-GPU; falls back to device_map="auto".
    """

    def _load_model_instance(self):
        from transformers import AutoModelForImageTextToText  # type: ignore
        return AutoModelForImageTextToText.from_pretrained(
            self.spec.path,
            dtype=self.spec.dtype,
            trust_remote_code=True,
            device_map="auto").eval()

# GLM-4.5V (HF)

class GLM45VStrategy(HuggingFaceStrategy):
    """
    GLM-4.5V (MoE) loader & generator.
    Uses the official Transformers class Glm4vMoeForConditionalGeneration.
    """
    def _load_model_instance(self):
        from transformers import Glm4vMoeForConditionalGeneration  # type: ignore
        return Glm4vMoeForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=self.spec.path,
            dtype="auto",   # bf16 recommended
            device_map="auto",
        ).eval()



# 5) vLLM local strategy

class VllmStrategy(ModelStrategy):
    """Runs a model through a local vLLM engine, sharded over all visible GPUs."""

    def load(self) -> None:
        assert LLM is not None, "vLLM package not installed"
        self._ensure_processor()

        self.model = LLM(
            model=self.spec.path,
            dtype="auto",
            trust_remote_code=True,
            tensor_parallel_size=torch.cuda.device_count(),
            gpu_memory_utilization=0.80,
            max_model_len=2**16,
        )
        self._device_for_inputs = "cpu"
        logger.info("vLLM model ready.")

    def generate(
        self,
        user_prompt: str,
        system_prompt: str,
        image: Optional[Union[Image.Image, Path]],
        gen_cfg: GenerationConfig,
        reference: str = "",
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:

        if history:
            raise NotImplementedError(
                f"[{self.spec.name}] history-conditioned generation is only "
                "implemented for the API strategies (Claude/Gemini/OpenAI)."
            )
        assert self.prompt_builder is not None, "vLLM prompt builder missing"

        print(f"Temp={gen_cfg.temperature}")

        req = self.prompt_builder.build_vllm_request(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            image=image if self.spec.supports_image else None,
        )
        # ensure we hand vLLM a PIL image, not a temp path
        if "multi_modal_data" in req and isinstance(req["multi_modal_data"].get("image"), str):
            req["multi_modal_data"]["image"] = _as_pil(image)

        params = SamplingParams(
            max_tokens=gen_cfg.max_new_tokens,
            temperature=0.0,
            top_p=1.0,
            stop_token_ids=[self.processor.tokenizer.eos_token_id],
        )
        out = self.model.generate([req], sampling_params=params, use_tqdm=False)[0]
        return out.outputs[0].text.strip()


# 6) Remote Gemini (API) strategy – optional, off GPU

class GeminiStrategy(ModelStrategy):
    """Calls the Gemini API via ``google-genai``."""

    def load(self) -> None:
        logger.info("[gemini] remote strategy initialized - will use API key from env")

    def generate(
        self,
        user_prompt: str,
        system_prompt: str,
        image: Optional[Union[Image.Image, Path]],
        gen_cfg: GenerationConfig,
        reference: str = "",
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        from google import genai
        from google.genai import types

        client = genai.Client()  # picks up GEMINI_API_KEY from env

        if history:
            from io import BytesIO

            def _turn_parts(text: str, img: Any) -> List[Any]:
                parts = [types.Part.from_text(text=text)]
                if img is not None:
                    buf = BytesIO()
                    _as_pil(img).save(buf, format="PNG")
                    parts.append(types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png"))
                return parts

            contents: List[Any] = [
                types.Content(
                    role="model" if turn["role"] == "assistant" else "user",
                    parts=_turn_parts(turn.get("text", ""), turn.get("image")),
                )
                for turn in history
            ]
            contents.append(types.Content(role="user", parts=_turn_parts(user_prompt, image)))
        else:
            contents = [user_prompt]
            if image is not None:
                contents.append(_as_pil(image))

        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            max_output_tokens=gen_cfg.max_new_tokens,
            temperature=gen_cfg.temperature,
            top_p=gen_cfg.top_p,
        )

        # NB: no pre-call count_tokens() — that was a second API request per step
        # (doubling usage against RPM limits). The real input token count is read
        # from the response's usage_metadata below.
        resp = client.models.generate_content(
            model=self.spec.path,
            contents=contents,
            config=config,
        )

        response_text = resp.text if resp.candidates else "Response was blocked."

        try:
            usage = resp.usage_metadata
            input_tokens = usage.prompt_token_count or 0
            output_tokens = usage.candidates_token_count or 0
            total_tokens = usage.total_token_count or 0
            # total - (input + output) captures thinking tokens for reasoning models
            thinking_tokens = total_tokens - (input_tokens + output_tokens)
            logger.info(
                "\n[gemini] Input tokens: %s"
                "\n[gemini] Estimated thinking tokens: %s"
                "\n[gemini] Estimated output tokens: %s",
                input_tokens,
                thinking_tokens,
                output_tokens,
            )
        except Exception as e:
            logger.warning("[gemini] Could not retrieve usage metadata from response. Error: %s", e)

        return response_text


# 6b) Remote OpenAI (API) strategy – optional, off GPU

class OpenAIStrategy(ModelStrategy):
    """Calls the OpenAI Responses/Chat API."""

    def load(self) -> None:
        logger.info("[openai] remote strategy initialized - will use API key from env")

    @staticmethod
    def _image_to_data_url(image: Union[Image.Image, Path, str]) -> str:
        import base64
        from io import BytesIO

        pil = _as_pil(image)
        buf = BytesIO()
        pil.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{b64}"

    def generate(
        self,
        user_prompt: str,
        system_prompt: str,
        image: Optional[Union[Image.Image, Path]],
        gen_cfg: GenerationConfig,
        reference: str = "",
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        from openai import OpenAI

        client = OpenAI()  # picks up OPENAI_API_KEY from env

        user_content: List[Dict[str, Any]] = [{"type": "text", "text": user_prompt}]
        if image is not None:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": self._image_to_data_url(image)},
            })

        messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        for turn in history or []:
            if turn["role"] == "assistant":
                messages.append({"role": "assistant", "content": turn.get("text", "")})
            else:
                content: List[Dict[str, Any]] = [{"type": "text", "text": turn.get("text", "")}]
                if turn.get("image") is not None:
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": self._image_to_data_url(turn["image"])},
                    })
                messages.append({"role": "user", "content": content})
        messages.append({"role": "user", "content": user_content})

        resp = client.chat.completions.create(
            model=self.spec.path,
            messages=messages,
            max_completion_tokens=gen_cfg.max_new_tokens,
            temperature=gen_cfg.temperature,
            top_p=gen_cfg.top_p,
        )

        try:
            usage = resp.usage
            prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
            completion_tokens = getattr(usage, "completion_tokens", 0) or 0
            total_tokens = getattr(usage, "total_tokens", 0) or 0
            reasoning_tokens = total_tokens - (prompt_tokens + completion_tokens)
            logger.info(
                "\n[openai] Input tokens: %s\n[openai] Output tokens: %s\n[openai] Reasoning/other tokens: %s",
                prompt_tokens,
                completion_tokens,
                reasoning_tokens,
            )
        except Exception as e:
            logger.warning("[openai] Could not retrieve usage metadata. Error: %s", e)

        choice = resp.choices[0] if resp.choices else None
        return choice.message.content if choice and choice.message else "Response was blocked."


# 6c) Remote Anthropic Claude (API) strategy – optional, off GPU

class ClaudeStrategy(ModelStrategy):
    """Calls the Anthropic Messages API."""

    # Per-model output-token caps from the Anthropic API.
    MAX_OUTPUT_TOKENS: Dict[str, int] = {
        "claude-opus-4-5": 32000,
        "claude-sonnet-4-5": 64000,
        "claude-haiku-4-5": 64000,
    }
    DEFAULT_MAX_OUTPUT_TOKENS = 32000

    def load(self) -> None:
        logger.info("[claude] remote strategy initialized - will use API key from env")

    @staticmethod
    def _image_to_b64(image: Union[Image.Image, Path, str]) -> str:
        import base64
        from io import BytesIO

        pil = _as_pil(image)
        buf = BytesIO()
        pil.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def generate(
        self,
        user_prompt: str,
        system_prompt: str,
        image: Optional[Union[Image.Image, Path]],
        gen_cfg: GenerationConfig,
        reference: str = "",
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        import anthropic

        client = anthropic.Anthropic()

        def _user_content(text: str, img: Any) -> List[Dict[str, Any]]:
            content: List[Dict[str, Any]] = []
            if img is not None:
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": self._image_to_b64(img),
                    },
                })
            content.append({"type": "text", "text": text})
            return content

        messages: List[Dict[str, Any]] = []
        for turn in history or []:
            if turn["role"] == "assistant":
                messages.append({"role": "assistant",
                                 "content": [{"type": "text", "text": turn.get("text", "")}]})
            else:
                messages.append({"role": "user",
                                 "content": _user_content(turn.get("text", ""), turn.get("image"))})
        messages.append({"role": "user", "content": _user_content(user_prompt, image)})

        # Anthropic rejects passing both temperature and top_p; send one.
        sampling_kwargs: Dict[str, Any] = {"temperature": gen_cfg.temperature}
        if gen_cfg.top_p != 1.0:
            sampling_kwargs = {"top_p": gen_cfg.top_p}

        cap = self.MAX_OUTPUT_TOKENS.get(self.spec.path, self.DEFAULT_MAX_OUTPUT_TOKENS)
        max_tokens = min(gen_cfg.max_new_tokens, cap)

        # Use streaming — required by SDK when max_tokens is large enough that
        # the request could exceed the 10-minute non-streaming timeout.
        with client.messages.stream(
            model=self.spec.path,
            system=system_prompt,
            messages=messages,
            max_tokens=max_tokens,
            # thinking={"type": "enabled", "budget_tokens": 10000},
            **sampling_kwargs,
        ) as stream:
            final = stream.get_final_message()

        try:
            usage = final.usage
            input_tokens = getattr(usage, "input_tokens", 0) or 0
            output_tokens = getattr(usage, "output_tokens", 0) or 0
            logger.info("\n[claude] Input tokens: %s\n[claude] Output tokens: %s", input_tokens, output_tokens)
        except Exception as e:
            logger.warning("[claude] Could not retrieve usage metadata. Error: %s", e)

        for block in final.content:
            if getattr(block, "type", None) == "text":
                return block.text
        return "Response was blocked."


# 7) Registry + factory

def get_strategy(name: str, engine: str, registry: Dict[str, ModelSpec]) -> ModelStrategy:
    """Resolve a registry *name* plus *engine* to a constructed strategy."""
    try:
        spec = registry[name]
    except KeyError as exc:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(registry)}") from exc

    if engine == "vllm":
        if spec.strategy_vllm is None:
            raise ValueError(f"Model '{name}' does not support vLLM backend")
        strat = spec.strategy_vllm(spec)
    elif engine == "hf":
        if spec.strategy_hf is None:
            raise ValueError(f"Model '{name}' does not support HuggingFace backend")
        strat = spec.strategy_hf(spec)
    else:
        raise ValueError(f"Unknown engine '{engine}', expected 'hf' or 'vllm'")

    strat.load()
    return strat


# 8) Assistant facade

class ModelAssistant:
    """Public entry point: resolves a backend name to a strategy and generates."""

    MODEL_REGISTRY: Dict[str, ModelSpec] = {
        "gemma3": ModelSpec(
            name="gemma3",
            path="google/gemma-3-27b-it",
            strategy_hf=GemmaStrategy,
            strategy_vllm=VllmStrategy,
        ),
        "gemma3-4b": ModelSpec(
            name="gemma3-4b",
            path="google/gemma-3-4b-it",
            strategy_hf=GemmaStrategy,
            strategy_vllm=VllmStrategy,
        ),
        "llama4": ModelSpec(
            name="llama4",
            path="meta-llama/Llama-4-Scout-17B-16E-Instruct",
            strategy_hf=LlamaStrategy,
            strategy_vllm=VllmStrategy,
        ),
        "qwen2.5-7b": ModelSpec(
            name="qwen2.5-7b",
            path="Qwen/Qwen2.5-VL-7B-Instruct",
            strategy_hf=QwenVLStrategy,
            strategy_vllm=VllmStrategy,
        ),
        "qwen2.5-32b": ModelSpec(
            name="qwen2.5-32b",
            path="Qwen/Qwen2.5-VL-32B-Instruct",
            strategy_hf=QwenVLStrategy,
            strategy_vllm=VllmStrategy,
        ),
        "qwen3-vl-thinking": ModelSpec(
            name="qwen3-vl-thinking",
            path="Qwen/Qwen3-VL-30B-A3B-Thinking",
            strategy_hf=QwenVLStrategy,
            strategy_vllm=VllmStrategy,
        ),
        "gemini2.5-pro": ModelSpec(
            name="gemini-2.5-pro",
            path="gemini-2.5-pro",
            strategy_hf=GeminiStrategy,
            strategy_vllm=GeminiStrategy,
            supports_image=True,
        ),
        "gemini3.1-pro": ModelSpec(
            name="gemini-3.1-pro",
            path="gemini-3.1-pro-preview",
            strategy_hf=GeminiStrategy,
            strategy_vllm=GeminiStrategy,
            supports_image=True,
        ),
        "gpt-5": ModelSpec(
            name="gpt-5",
            path="gpt-5",
            strategy_hf=OpenAIStrategy,
            strategy_vllm=OpenAIStrategy,
            supports_image=True,
        ),
        "gpt-5-mini": ModelSpec(
            name="gpt-5-mini",
            path="gpt-5-mini",
            strategy_hf=OpenAIStrategy,
            strategy_vllm=OpenAIStrategy,
            supports_image=True,
        ),
        "gpt-4o": ModelSpec(
            name="gpt-4o",
            path="gpt-4o",
            strategy_hf=OpenAIStrategy,
            strategy_vllm=OpenAIStrategy,
            supports_image=True,
        ),
        "claude-opus-4.5": ModelSpec(
            name="claude-opus-4.5",
            path="claude-opus-4-5",
            strategy_hf=ClaudeStrategy,
            strategy_vllm=ClaudeStrategy,
            supports_image=True,
        ),
        "claude-sonnet-4.5": ModelSpec(
            name="claude-sonnet-4.5",
            path="claude-sonnet-4-5",
            strategy_hf=ClaudeStrategy,
            strategy_vllm=ClaudeStrategy,
            supports_image=True,
        ),
        "claude-haiku-4.5": ModelSpec(
            name="claude-haiku-4.5",
            path="claude-haiku-4-5",
            strategy_hf=ClaudeStrategy,
            strategy_vllm=ClaudeStrategy,
            supports_image=True,
        ),
        "internvl3_5-38b": ModelSpec(
            name="internvl3_5-38b",
            path="OpenGVLab/InternVL3_5-38B",
            strategy_hf=InternVL3_5Strategy,
            strategy_vllm=None,
            dtype=torch.bfloat16,
            supports_image=True,
        ),
        "glm4.5v": ModelSpec(
            name="glm4.5v",
            path="Zai/GLM-4.5V",
            strategy_hf=GLM45VStrategy,
            strategy_vllm=VllmStrategy,
            dtype=torch.bfloat16,
            supports_image=True,
        ),
    }

    def __init__(self, backend: str, engine: str) -> None:
        logger.info("Initializing assistant backend=%s engine=%s", backend, engine)
        self.backend = backend
        self.engine = engine
        if backend not in self.MODEL_REGISTRY:
            raise ValueError(f"Unknown model '{backend}'")
        self.strategy = get_strategy(backend, self.engine, self.MODEL_REGISTRY)

    def get_name(self) -> str:
        """The registry name of the backend in use."""
        return self.MODEL_REGISTRY[self.backend].name

    def generate(
        self,
        user_prompt: str,
        system_prompt: str = "You are a helpful assistant.",
        *,
        max_new_tokens: int = 128,
        image: Optional[Union[Image.Image, Path]] = None,
        reference: str = "",
        temperature: float = 0.0,
        top_p: float = 1.0,
        do_sample: bool = False,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Generate one completion from the active backend.

        ``history`` prepends prior turns before the final (user_prompt, image) turn
        and is only supported by the API strategies.
        """
        gen_cfg = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
        )
        # Retry transient API failures (rate limits, timeouts, 5xx) so a single
        # hiccup doesn't kill a long concurrent run. Non-transient errors (bad
        # request, auth, local-model failures) re-raise immediately.
        return _call_with_retry(
            lambda: self.strategy.generate(
                user_prompt, system_prompt, image, gen_cfg, reference, history=history
            ),
            label=self.get_name(),
        )

    def cleanup(self) -> None:
        """Release the backend's resources."""
        self.strategy.cleanup()


# 9) Prompt-file helper (unchanged)


@lru_cache(maxsize=1)
def load_prompts() -> Dict[str, Any]:
    """Load ./prompts.yaml once and cache it."""
    path = Path("prompts.yaml")
    if not path.exists():
        raise FileNotFoundError("prompts.yaml file not found")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# 10) Tiny demo


def _demo() -> None:  # pragma: no cover
    """Ask one configured backend to describe a freshly created cube image."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemma3", help="Backend model name")
    parser.add_argument("--engine", default="hf", choices=["hf", "vllm"], help="Execution engine")
    args = parser.parse_args()

    backend = args.model
    engine = args.engine

    cube = VirtualCube()

    assistant = ModelAssistant(backend, engine=engine)
    try:
        print(f"\n=========== Response (backend: {backend}, engine: {engine}) ===========")
        out = assistant.generate(
            system_prompt="You are a good assistant.",
            user_prompt="Describe me the image you see. What are the colors on the front face?",
            image=cube.to_image(),
            max_new_tokens=2**12,
        )
        print(out)
    finally:
        assistant.cleanup()


if __name__ == "__main__":  # pragma: no cover
    _demo()
