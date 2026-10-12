from __future__ import annotations

import os
import re
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam

from .subtitles import SubtitleSegment, bilingual_segments, read_subtitles, write_subtitles


class TranslationError(RuntimeError):
    pass


DEFAULT_GLOSSARY: tuple[tuple[str, str, str], ...] = (
    ("アオイちゃん", "葵酱", "角色名或称呼"),
    ("葵ちゃん", "葵酱", "角色名或称呼"),
    ("葵くん", "葵君", "角色名或称呼"),
)


@dataclass(frozen=True)
class SakuraConfig:
    base_url: str
    api_key: str
    model: str
    temperature: float = 0.3
    top_p: float = 0.8

    @classmethod
    def from_env(
        cls,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float = 0.3,
        top_p: float = 0.8,
    ) -> SakuraConfig:
        resolved_base_url = base_url or os.environ.get("SAKURA_BASE_URL")
        resolved_api_key = api_key or os.environ.get("SAKURA_API_KEY", "not-needed")
        resolved_model = model or os.environ.get("SAKURA_MODEL")
        if not resolved_base_url:
            raise TranslationError(
                "missing Sakura API base URL; set SAKURA_BASE_URL or pass base_url"
            )
        if not resolved_model:
            raise TranslationError("missing Sakura model; set SAKURA_MODEL or pass model")
        return cls(
            base_url=resolved_base_url,
            api_key=resolved_api_key,
            model=resolved_model,
            temperature=temperature,
            top_p=top_p,
        )


def translate_file(
    input_path: Path,
    output_path: Path,
    *,
    config: SakuraConfig,
    batch_size: int = 20,
    bilingual_output_path: Path | None = None,
    source_first: bool = True,
) -> None:
    source_segments = read_subtitles(input_path)
    translated_segments = translate_segments(source_segments, config=config, batch_size=batch_size)
    write_subtitles(translated_segments, output_path)

    if bilingual_output_path is not None:
        bilingual = bilingual_segments(
            source_segments,
            translated_segments,
            source_first=source_first,
        )
        write_subtitles(bilingual, bilingual_output_path)


def translate_segments(
    segments: list[SubtitleSegment],
    *,
    config: SakuraConfig,
    batch_size: int = 20,
    complete: Callable[[list[ChatCompletionMessageParam]], str] | None = None,
) -> list[SubtitleSegment]:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    client = (
        None
        if complete
        else OpenAI(base_url=config.base_url, api_key=config.api_key, timeout=3600, max_retries=0)
    )
    try:
        translated: list[SubtitleSegment] = []
        for start in range(0, len(segments), batch_size):
            batch = segments[start : start + batch_size]
            translated.extend(
                _translate_batch(
                    batch,
                    client=client,
                    config=config,
                    complete=complete,
                )
            )
        return translated
    finally:
        if client is not None:
            client.close()


def _translate_batch(
    batch: list[SubtitleSegment],
    *,
    client: OpenAI | None,
    config: SakuraConfig,
    complete: Callable[[list[ChatCompletionMessageParam]], str] | None,
) -> list[SubtitleSegment]:
    messages = _build_messages(batch)
    raw_response = (
        complete(messages) if complete else _complete_chat(client, config=config, messages=messages)
    )
    try:
        texts = parse_translation_response(
            raw_response,
            expected_ids=range(1, len(batch) + 1),
        )
    except TranslationError:
        if len(batch) == 1:
            raise
        midpoint = len(batch) // 2
        return [
            *_translate_batch(
                batch[:midpoint],
                client=client,
                config=config,
                complete=complete,
            ),
            *_translate_batch(
                batch[midpoint:],
                client=client,
                config=config,
                complete=complete,
            ),
        ]

    return [
        SubtitleSegment(
            index=segment.index,
            start=segment.start,
            end=segment.end,
            text=text.strip(),
        )
        for segment, text in zip(batch, texts, strict=True)
    ]


def parse_translation_response(
    raw_response: str,
    *,
    expected_ids: Sequence[int],
) -> list[str]:
    expected = list(expected_ids)
    if not expected:
        return []

    blocks = _parse_numbered_blocks(_strip_code_fence(raw_response))
    translations_by_id: dict[int, str] = {}
    for block_id, text in blocks:
        if block_id in translations_by_id:
            raise TranslationError(f"Sakura returned duplicate translation block [{block_id}]")
        translations_by_id[block_id] = text

    expected_set = set(expected)
    missing = [block_id for block_id in expected if block_id not in translations_by_id]
    extra = [block_id for block_id in translations_by_id if block_id not in expected_set]
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing {missing}")
        if extra:
            details.append(f"unexpected {extra}")
        raise TranslationError(
            f"Sakura response block ids did not match input: {', '.join(details)}"
        )

    return [translations_by_id[block_id].strip() for block_id in expected]


def default_translated_path(input_path: Path) -> Path:
    return _replace_language_suffix(input_path, "zh")


def default_bilingual_path(input_path: Path) -> Path:
    return _replace_language_suffix(input_path, "ja-zh")


def _build_messages(batch: list[SubtitleSegment]) -> list[ChatCompletionMessageParam]:
    return [
        {
            "role": "system",
            "content": (
                "You are Sakura, a Japanese-to-Simplified-Chinese subtitle translator. "
                "你是一个视觉小说翻译模型，可以通顺地使用给定术语表，"
                "以自然流畅的风格将日文翻译成简体中文。不要擅自添加原文没有的内容，"
                "不要增加或减少换行。"
            ),
        },
        {
            "role": "user",
            "content": (
                "参考以下术语表（可为空，格式为 src->dst #备注）：\n"
                f"{_format_glossary(DEFAULT_GLOSSARY)}\n\n"
                "根据以上术语表的对应关系和备注，结合上下文，"
                "将下面的文本从日文翻译成简体中文。\n"
                "请保留每个 [数字] 编号，只翻译编号下面的正文。"
                "不要输出解释，不要输出 JSON。\n"
                "[Input]\n"
                f"{_format_numbered_blocks(batch)}"
            ),
        },
    ]


_BLOCK_HEADER_RE = re.compile(r"^\s*\[(?P<id>\d+)]\s*(?P<text>.*)$")


def _format_numbered_blocks(batch: list[SubtitleSegment]) -> str:
    return "\n\n".join(
        f"[{block_id}]\n{segment.text.strip()}" for block_id, segment in enumerate(batch, start=1)
    )


def _format_glossary(glossary: tuple[tuple[str, str, str], ...]) -> str:
    return "\n".join(f"{source}->{target} #{note}" for source, target, note in glossary)


def _parse_numbered_blocks(text: str) -> list[tuple[int, str]]:
    blocks: list[tuple[int, str]] = []
    current_id: int | None = None
    current_lines: list[str] = []

    for line in text.splitlines():
        match = _BLOCK_HEADER_RE.match(line)
        if match:
            if current_id is not None:
                blocks.append((current_id, _clean_block_text(current_lines)))
            current_id = int(match.group("id"))
            inline_text = match.group("text").strip()
            current_lines = [inline_text] if inline_text else []
            continue

        if current_id is None:
            continue
        current_lines.append(line)

    if current_id is not None:
        blocks.append((current_id, _clean_block_text(current_lines)))

    if not blocks:
        raise TranslationError("Sakura response did not contain numbered translation blocks")
    return blocks


def _clean_block_text(lines: list[str]) -> str:
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines).strip()


def _complete_chat(
    client: OpenAI | None,
    *,
    config: SakuraConfig,
    messages: list[ChatCompletionMessageParam],
) -> str:
    if client is None:
        raise TranslationError("OpenAI client was not initialized")
    raw = client.chat.completions.with_raw_response.create(
        model=config.model,
        messages=messages,
        temperature=config.temperature,
        top_p=config.top_p,
        stream=False,
    )
    warning = raw.headers.get("x-local-llm-warning")
    if warning:
        warnings.warn(f"local LLM: {warning}", RuntimeWarning, stacklevel=2)
    response = raw.parse()
    if not isinstance(response, ChatCompletion):
        raise TranslationError("Sakura returned a streaming response to a non-streaming request")
    content = response.choices[0].message.content
    if not content:
        raise TranslationError("Sakura returned an empty response")
    return content


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if len(lines) >= 3 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return stripped


def _replace_language_suffix(input_path: Path, language: str) -> Path:
    subtitle_suffix = input_path.suffix
    stem = input_path.stem
    if stem.endswith(".ja"):
        stem = stem.removesuffix(".ja")
    elif stem.endswith(".zh") or stem.endswith(".ja-zh"):
        stem = stem.rsplit(".", 1)[0]
    return input_path.with_name(f"{stem}.{language}{subtitle_suffix}")
