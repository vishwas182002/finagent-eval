"""Qwen2.5-VL adapter - the same prompt, quantization and decoding settings as
the Colab notebook that produced ``results/track_a/qwen25vl_7b_results.json``.

Requires a CUDA GPU and the optional ``vlm`` extra::

    pip install "finagent-eval[vlm]"
    finagent-eval run --adapter qwen2.5-vl --out results/track_a/qwen25vl_7b_rerun.json
"""
from __future__ import annotations

from pathlib import Path

from .base import BaseAdapter

DEFAULT_PROMPT = (
    "Answer the question using only the document image. "
    "Return only the final answer, with no explanation.\n\n"
    "Question: {question}"
)


class QwenVLAdapter(BaseAdapter):
    name = "Qwen2.5-VL-7B-Instruct"

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        load_in_4bit: bool = True,
        max_new_tokens: int = 64,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1024 * 28 * 28,
        prompt_template: str = DEFAULT_PROMPT,
        device: str = "cuda",
        oom_retry_max_new_tokens: int | None = 32,
        revision: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.load_in_4bit = load_in_4bit
        self.max_new_tokens = max_new_tokens
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.prompt_template = prompt_template
        self.device = device
        self.oom_retry_max_new_tokens = oom_retry_max_new_tokens
        self.revision = revision  # pin to a Hugging Face commit sha for a frozen run
        self.model = None
        self.processor = None

    def setup(self) -> None:
        try:
            import torch
            from qwen_vl_utils import process_vision_info  # noqa: F401 - checked here so failures are early
            from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
        except ImportError as exc:  # pragma: no cover - depends on optional deps
            raise ImportError(
                "QwenVLAdapter needs torch, transformers, bitsandbytes, accelerate and qwen-vl-utils. "
                "Install with: pip install 'finagent-eval[vlm]'"
            ) from exc

        quant = None
        if self.load_in_4bit:
            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id, quantization_config=quant, device_map="auto", revision=self.revision
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_id, min_pixels=self.min_pixels, max_pixels=self.max_pixels, revision=self.revision
        )

    def predict(self, image_path: Path, question: str) -> str:  # pragma: no cover - needs GPU
        import gc

        import torch

        try:
            return self._generate(image_path, question, self.max_new_tokens)
        except torch.cuda.OutOfMemoryError:
            if self.oom_retry_max_new_tokens is None:
                raise
            # Same recovery as the original notebook run: clear the cache and retry shorter.
            torch.cuda.empty_cache()
            gc.collect()
            return self._generate(image_path, question, self.oom_retry_max_new_tokens)

    def _generate(self, image_path: Path, question: str, max_new_tokens: int) -> str:  # pragma: no cover - needs GPU
        import torch
        from qwen_vl_utils import process_vision_info

        if self.model is None:
            self.setup()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": self.prompt_template.format(question=question)},
                ],
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
        ).to(self.device)

        with torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated, strict=True)]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

    def metadata(self) -> dict:
        return {
            "model_id": self.model_id,
            "quantization": "bitsandbytes nf4, double-quant, fp16 compute" if self.load_in_4bit else "none",
            "revision": self.revision,
            "decoding": {
                "do_sample": False,
                "max_new_tokens": self.max_new_tokens,
                "oom_retry_max_new_tokens": self.oom_retry_max_new_tokens,
            },
            "image_pixels": {"min": self.min_pixels, "max": self.max_pixels},
            "prompt_template": self.prompt_template,
        }
