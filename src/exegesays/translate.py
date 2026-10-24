"""Chinese-to-English translation with Hy-MT2 on Apple Silicon."""

from __future__ import annotations

import copy
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.generate import generate_step
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import greedy_sampler

from exegesays.context import (
    HISTORY_LIMIT,
    looks_context_dependent,
    should_retry_with_context,
)

logging.getLogger("transformers").setLevel(logging.ERROR)


@dataclass(frozen=True)
class Translation:
    text: str
    mean_logprob: float
    used_context: bool


class Translator:
    def __init__(self, model_dir: Path, target: str = "en") -> None:
        if target not in {"en", "zh"}:
            raise ValueError(f"Unsupported target language: {target}")
        self._target = target
        self._keep = "zh" if target == "en" else "en"
        self._model, self._tokenizer = load(str(model_dir))
        self._instruction = (
            f"将以下文本翻译为{self._target_name}，注意只需要输出翻译后的结果，不要额外解释："
        )
        self._context_instruction = (
            f"请结合背景信息将以下文本翻译为{self._target_name}。"
        )
        self._source_heading = "【待翻译文本】"
        self._prefix_tokens, self._prefix_cache = self._warm_prefix_cache()

    @property
    def _target_name(self) -> str:
        return "英语" if self._target == "en" else "中文"

    def _warm_prefix_cache(self) -> tuple[list[int], list | None]:
        first = _encode_prompt(self._tokenizer, self._prompt("AAAA"))
        second = _encode_prompt(self._tokenizer, self._prompt("BBBB"))
        prefix_len = _shared_prefix_len(first, second)
        prefix_len = min(prefix_len, len(first) - 1, len(second) - 1)
        if prefix_len < 8:
            return [], None
        tokens = first[:prefix_len]
        cache = make_prompt_cache(self._model)
        for _ in generate_step(
            mx.array(tokens),
            self._model,
            max_tokens=0,
            prompt_cache=cache,
            sampler=greedy_sampler,
        ):
            pass
        mx.eval([item.state for item in cache])
        return tokens, cache

    def translate(
        self,
        text: str,
        history: list[tuple[str, str]] | None = None,
        on_token: Callable[[str], None] | None = None,
        on_reset: Callable[[], None] | None = None,
    ) -> Translation:
        source = text.strip()
        if not source:
            return Translation("", 0.0, False)
        prior = (history or [])[-HISTORY_LIMIT:]
        stream_first = on_token is not None and not (
            prior and looks_context_dependent(source, self._keep)
        )
        first, mean_logprob = self._run(
            source, on_token=on_token if stream_first else None
        )
        if not first:
            return Translation("", mean_logprob, False)
        if should_retry_with_context(
            source, mean_logprob, bool(prior), keep=self._keep
        ):
            if stream_first and on_reset is not None:
                on_reset()
            retry, retry_lp = self._run(source, prior, on_token=on_token)
            if retry:
                return Translation(retry, retry_lp, True)
        return Translation(first, mean_logprob, False)

    def _run(
        self,
        source: str,
        history: list[tuple[str, str]] | None = None,
        on_token: Callable[[str], None] | None = None,
    ) -> tuple[str, float]:
        prompt = self._prompt(source, history)
        prompt_arg: str | list[int] = prompt
        prompt_cache = None
        if not history and self._prefix_cache is not None:
            tokens = _encode_prompt(self._tokenizer, prompt)
            prefix = self._prefix_tokens
            if len(tokens) > len(prefix) and tokens[: len(prefix)] == prefix:
                prompt_arg = tokens[len(prefix) :]
                prompt_cache = copy.deepcopy(self._prefix_cache)
        pieces: list[str] = []
        token_logprobs: list[float] = []
        for response in stream_generate(
            self._model,
            self._tokenizer,
            prompt=prompt_arg,
            max_tokens=256,
            sampler=greedy_sampler,
            prompt_cache=prompt_cache,
        ):
            piece = response.text or ""
            pieces.append(piece)
            if piece and on_token is not None:
                on_token(piece)
            if response.finish_reason is not None:
                continue
            try:
                token_logprobs.append(
                    float(response.logprobs[response.token].item())
                )
            except Exception:
                pass
        mean_logprob = (
            sum(token_logprobs) / len(token_logprobs) if token_logprobs else 0.0
        )
        return (
            _clean_translation(
                "".join(pieces),
                source,
                self._instruction,
                self._context_instruction,
                self._source_heading,
            ),
            mean_logprob,
        )

    def _prompt(
        self,
        source: str,
        history: list[tuple[str, str]] | None = None,
    ) -> str:
        if history:
            lines: list[str] = []
            for source_line, translated in history:
                lines.append(source_line)
                if translated:
                    lines.append(translated)
            message = (
                "【背景信息】\n"
                + "\n".join(lines)
                + "\n\n"
                + self._context_instruction
                + "\n\n"
                + f"{self._source_heading}\n"
                + source
            )
        else:
            message = f"{self._instruction}\n\n{source}"
        messages = [{"role": "user", "content": message}]
        tokenizer = self._tokenizer
        if getattr(tokenizer, "chat_template", None):
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return message


def _encode_prompt(tokenizer, prompt: str) -> list[int]:
    bos = getattr(tokenizer, "bos_token", None)
    add_special_tokens = bos is None or not prompt.startswith(bos)
    tokens = tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
    if hasattr(tokens, "tolist"):
        tokens = tokens.tolist()
    return list(tokens)


def _shared_prefix_len(left: list[int], right: list[int]) -> int:
    length = 0
    for first, second in zip(left, right):
        if first != second:
            break
        length += 1
    return length


def _clean_translation(
    response: str,
    source: str,
    instruction: str,
    context_instruction: str,
    source_heading: str = "【待翻译文本】",
) -> str:
    text = response.strip()
    for marker in (source_heading, instruction, context_instruction):
        if marker in text:
            text = text.split(marker, 1)[1].strip()
    if source and source in text:
        text = text.replace(source, "", 1).strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines and lines[0].lower() in {"assistant", "assistant:"}:
        lines = lines[1:]
    return " ".join(lines).strip()
