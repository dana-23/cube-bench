"""Model strategies for local HuggingFace/vLLM and hosted API backends."""

from __future__ import annotations

import base64
import logging
import os
import random
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Type, Union

import torch
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

# Remote API retry policy
_RETRY_MAX = int(os.getenv("CUBE_BENCH_API_MAX_RETRIES", "5"))
_RETRY_BASE_SEC = float(os.getenv("CUBE_BENCH_API_RETRY_BASE_SEC", "2.0"))
_RETRY_CAP_SEC = float(os.getenv("CUBE_BENCH_API_RETRY_CAP_SEC", "60.0"))

# Retry rate limits, timeouts, and transient server errors (including Anthropic 529).
_TRANSIENT_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}
# Cross-SDK exception cues avoid importing provider packages here.
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
    """Share a rolling ``rpm`` limit across threads; non-positive values disable it."""

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
            time.sleep(max(0.0, sleep_for) + 0.001)  # Never hold the lock while sleeping.


# Optional client-side request cap; zero is unlimited.
_RPM_LIMIT = int(os.getenv("CUBE_BENCH_API_RPM", "0"))
_RATE_LIMITER = _RateLimiter(_RPM_LIMIT)


def _call_with_retry(fn, label: str):
    """Rate-limit ``fn`` and retry transient failures with backoff and jitter."""
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


# Optional local-model dependencies.
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

# Keep vLLM quiet without suppressing application logs.
logging.getLogger("vllm").setLevel(logging.WARNING)
logging.getLogger("vllm.core").setLevel(logging.WARNING)


# Shared data structures and utilities

@dataclass(frozen=True)
class ModelSpec:
    """One registry entry: where a model lives and which strategy runs it."""

    name: str
    path: str
    strategy_hf: Optional[Type["ModelStrategy"]] = None
    strategy_vllm: Optional[Type["ModelStrategy"]] = None
    dtype: torch.dtype = torch.bfloat16
    supports_image: bool = True

@dataclass
class GenerationConfig:
    """Decoding parameters shared by every strategy."""

    max_new_tokens: int = 256
    thinking_budget: Optional[int] = None
    temperature: float = 0.0
    top_p: float = 1.0
    do_sample: bool = True

def _as_pil(img: Optional[Union[Image.Image, Path, str]]) -> Optional[Image.Image]:
    """Normalize a single image input to a RGB PIL.Image (or None)."""
    if img is None:
        return None
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    return Image.open(str(img)).convert("RGB")


def _image_to_png_bytes(img: Union[Image.Image, Path, str]) -> bytes:
    """Encode a single image input as PNG bytes."""
    buf = BytesIO()
    _as_pil(img).save(buf, format="PNG")
    return buf.getvalue()


def _to_device(batch: Any, device: torch.device, dtype: Optional[torch.dtype] = None) -> Any:
    # HF BatchFeature has its own ``to``; plain dictionaries need field-wise moves.
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


# Prompt construction

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

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": ([{"type": "image", "image": pil}] if pil is not None else []) +
                                    [{"type": "text", "text": user_prompt}]},
        ]
        if reference:
            messages.append({"role": "assistant", "content": [{"type": "text", "text": reference}]})

        # Prefer native multimodal templates, falling back for text-only tokenizers.
        try:
            return self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                return_dict=True,
            )
        except TypeError as e:
            if "concatenate str (not \"list\") to str" not in str(e):
                raise

            tok = getattr(self.processor, "tokenizer", self.processor)
            image_processor = getattr(self, "image_processor", getattr(self.processor, "image_processor", None))

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

            text_inputs = tok(chat_text, return_tensors="pt", padding=True)

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
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",
            "content": ([{"type": "image"}] if pil is not None else []) + [{"type": "text", "text": user_prompt}]},
        ]
        prompt_txt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        req = {"prompt": prompt_txt}
        if pil is not None:
            req["multi_modal_data"] = {"image": pil}
        return req



# Strategy base class

class ModelStrategy(ABC):
    """Backend contract: load a model, generate from it, then release it."""

    def __init__(self, spec: ModelSpec):
        self.spec = spec
        self.processor: Optional["AutoProcessor"] = None
        self.model: Any = None
        self.prompt_builder: Optional[PromptBuilder] = None
        self._device_for_inputs: str = "cpu"

    # Lifecycle
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
        """Generate a response, optionally prepending API-only conversation history."""

    # Helpers
    def _reject_history(self, history: Optional[List[Dict[str, Any]]]) -> None:
        """Local strategies cannot condition on prior turns."""
        if history:
            raise NotImplementedError(
                f"[{self.spec.name}] history-conditioned generation is only "
                "implemented for the API strategies (Claude/Gemini/OpenAI)."
            )

    def _ensure_processor(self) -> None:
        assert AutoProcessor is not None, "transformers not installed"
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
            )

        self.prompt_builder = PromptBuilder(self.processor)

    def cleanup(self) -> None:
        """Drop references to the model, processor and prompt builder."""
        logger.debug("[%s] cleanup", self.spec.name)
        self.model = self.processor = self.prompt_builder = None  # type: ignore
        torch.cuda.empty_cache()


# HuggingFace strategies

class HuggingFaceStrategy(ModelStrategy):
    """Runs a model in-process through Transformers."""

    def load(self) -> None:
        self._ensure_processor()
        self.model = self._load_model_instance()

        # Sharded models lack one device, so stage inputs on the first accelerator.
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

        self._reject_history(history)
        assert self.prompt_builder is not None and self.processor is not None
        inputs = self.prompt_builder.build_hf_inputs(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            image=image if self.spec.supports_image else None,
            reference=reference,
        )

        # BatchFeature.to() ignores dtype, so move first and cast vision inputs below.
        inputs = _to_device(inputs, torch.device(self._device_for_inputs), dtype=None)

        # Match the vision tower dtype to avoid float/bfloat16 convolution errors.
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

        gen_ids = out[:, input_len:]
        text = self.processor.decode(gen_ids[0], skip_special_tokens=True)
        return text

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

# Concrete HuggingFace models

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
        self._reject_history(history)

        content = []
        if image is not None:
            if isinstance(image, (str, Path)):
                content.append({"type": "image", "image": str(image)})
            else:
                pil = image if image.mode == "RGB" else image.convert("RGB")
                content.append({"type": "image", "image": pil})
        content.append({"type": "text", "text": user_prompt})

        messages = [{"role": "user", "content": content}]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        device = getattr(self.model, "device", torch.device("cuda"))
        inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}

        max_new = int(getattr(gen_cfg, "max_new_tokens", 128))
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return output_text[0] if output_text else ""


class InternVL3_5Strategy(HuggingFaceStrategy):
    """InternVL3 loader using automatic device mapping."""

    def _load_model_instance(self):
        from transformers import AutoModelForImageTextToText  # type: ignore
        return AutoModelForImageTextToText.from_pretrained(
            self.spec.path,
            dtype=self.spec.dtype,
            trust_remote_code=True,
            device_map="auto").eval()

class GLM45VStrategy(HuggingFaceStrategy):
    """GLM-4.5V MoE loader using its Transformers model class."""
    def _load_model_instance(self):
        from transformers import Glm4vMoeForConditionalGeneration  # type: ignore
        return Glm4vMoeForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=self.spec.path,
            dtype="auto",
            device_map="auto",
        ).eval()



# vLLM strategy

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

        self._reject_history(history)
        assert self.prompt_builder is not None, "vLLM prompt builder missing"

        print(f"Temp={gen_cfg.temperature}")

        req = self.prompt_builder.build_vllm_request(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            image=image if self.spec.supports_image else None,
        )
        # vLLM accepts a PIL image directly.
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


# Hosted API strategies

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

        client = genai.Client()

        if history:
            def _turn_parts(text: str, img: Any) -> List[Any]:
                parts = [types.Part.from_text(text=text)]
                if img is not None:
                    parts.append(
                        types.Part.from_bytes(data=_image_to_png_bytes(img), mime_type="image/png")
                    )
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

        config_kwargs = {
            "system_instruction": system_prompt,
            "max_output_tokens": gen_cfg.max_new_tokens,
            "temperature": gen_cfg.temperature,
            "top_p": gen_cfg.top_p,
        }
        if gen_cfg.thinking_budget is not None:
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_budget=gen_cfg.thinking_budget
            )
        config = types.GenerateContentConfig(**config_kwargs)

        # Read usage from the response; a preflight count would consume another RPM slot.
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
            thinking_tokens = usage.thoughts_token_count
            if thinking_tokens is None:
                thinking_tokens = total_tokens - (input_tokens + output_tokens)
            logger.info(
                "\n[gemini] Input tokens: %s"
                "\n[gemini] Thinking tokens: %s"
                "\n[gemini] Estimated output tokens: %s",
                input_tokens,
                thinking_tokens,
                output_tokens,
            )
        except Exception as e:
            logger.warning("[gemini] Could not retrieve usage metadata from response. Error: %s", e)

        return response_text


class OpenAIStrategy(ModelStrategy):
    """Calls the OpenAI Responses/Chat API."""

    def load(self) -> None:
        logger.info("[openai] remote strategy initialized - will use API key from env")

    @staticmethod
    def _image_to_data_url(image: Union[Image.Image, Path, str]) -> str:
        b64 = base64.b64encode(_image_to_png_bytes(image)).decode("ascii")
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

        client = OpenAI()

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


class ClaudeStrategy(ModelStrategy):
    """Calls the Anthropic Messages API."""

    # Anthropic enforces model-specific output-token caps.
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
        return base64.b64encode(_image_to_png_bytes(image)).decode("ascii")

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

        # Anthropic rejects requests that specify both temperature and top_p.
        sampling_kwargs: Dict[str, Any] = {"temperature": gen_cfg.temperature}
        if gen_cfg.top_p != 1.0:
            sampling_kwargs = {"top_p": gen_cfg.top_p}

        cap = self.MAX_OUTPUT_TOKENS.get(self.spec.path, self.DEFAULT_MAX_OUTPUT_TOKENS)
        max_tokens = min(gen_cfg.max_new_tokens, cap)

        # Large outputs require streaming to avoid the SDK's non-streaming timeout.
        with client.messages.stream(
            model=self.spec.path,
            system=system_prompt,
            messages=messages,
            max_tokens=max_tokens,
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


# Registry and factory

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


# Assistant facade

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
        thinking_budget: Optional[int] = None,
        image: Optional[Union[Image.Image, Path]] = None,
        reference: str = "",
        temperature: float = 0.0,
        top_p: float = 1.0,
        do_sample: bool = False,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Generate a completion, optionally with API-only prior-turn history."""
        gen_cfg = GenerationConfig(
            max_new_tokens=max_new_tokens,
            thinking_budget=thinking_budget,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
        )
        # Retry only transient provider failures; local and permanent errors re-raise.
        return _call_with_retry(
            lambda: self.strategy.generate(
                user_prompt, system_prompt, image, gen_cfg, reference, history=history
            ),
            label=self.get_name(),
        )

    def cleanup(self) -> None:
        """Release the backend's resources."""
        self.strategy.cleanup()


# Demo
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
