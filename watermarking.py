from __future__ import annotations

import contextlib
import sys
from pathlib import Path
from typing import Any

from transformers import LogitsProcessorList


def _project_root() -> Path:
    here = Path(__file__).resolve().parent
    for root in [here, here.parent]:
        if (root / "MarkLLM").exists():
            return root
    return here.parent


def _kgw_config_path() -> Path:
    return _project_root() / "MarkLLM" / "config" / "KGW.json"


@contextlib.contextmanager
def _markllm_path():
    root = _project_root().resolve()
    markllm_root = root / "MarkLLM"
    original_path = list(sys.path)
    original_watermark = sys.modules.pop("watermark", None)
    sys.path = [
        str(markllm_root),
        *[path for path in sys.path if path and Path(path).resolve() != root],
    ]
    try:
        yield
    finally:
        sys.path = original_path
        if original_watermark is not None:
            sys.modules["watermark"] = original_watermark
        else:
            sys.modules.pop("watermark", None)


def _load_kgw(tokenizer: Any, model: Any, config_path: Path, device: str):
    with _markllm_path():
        from utils.transformers_config import TransformersConfig
        from watermark.kgw import KGW as MarkLLMKGW

    config = TransformersConfig(
        model=model,
        tokenizer=tokenizer,
        vocab_size=len(tokenizer),
        device=device,
    )
    return MarkLLMKGW(str(config_path), config)


class KGW:
    def __init__(self, model, algorithm_config: str | Path | None = None):
        self.generator = model
        self.tokenizer = self.generator.tokenizer
        self.device = str(self.generator.model.device)
        self.watermark = _load_kgw(
            tokenizer=self.tokenizer,
            model=self.generator.model,
            config_path=Path(algorithm_config) if algorithm_config else _kgw_config_path(),
            device=self.device,
        )

    @property
    def logits_processor(self) -> LogitsProcessorList:
        return LogitsProcessorList([self.watermark.logits_processor])

    def generate(self, prompt: str, *, enable_thinking: bool = False) -> str:
        return self.generator.generate(
            prompt,
            logits_processor=self.logits_processor,
            enable_thinking=enable_thinking,
        )

    def detect(self, text: str, *, return_dict: bool = True, return_green_flags: bool = False):
        return self.watermark.detect_watermark(
            text,
            return_dict=return_dict,
            return_green_flags=return_green_flags,
        )

