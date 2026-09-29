from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import srt


@dataclass(frozen=True)
class SubtitleSegment:
    index: int
    start: float
    end: float
    text: str


def read_subtitles(path: Path) -> list[SubtitleSegment]:
    if path.suffix.lower() != ".srt":
        raise ValueError(f"unsupported subtitle format: {path.suffix}")
    return read_srt(path)


def write_subtitles(segments: list[SubtitleSegment], path: Path) -> None:
    if path.suffix.lower() != ".srt":
        raise ValueError(f"unsupported subtitle format: {path.suffix}")
    write_srt(segments, path)


def read_srt(path: Path) -> list[SubtitleSegment]:
    return [
        SubtitleSegment(
            index=int(subtitle.index),
            start=subtitle.start.total_seconds(),
            end=subtitle.end.total_seconds(),
            text=subtitle.content.strip(),
        )
        for subtitle in srt.parse(path.read_text(encoding="utf-8"))
        if subtitle.content.strip()
    ]


def write_srt(segments: list[SubtitleSegment], path: Path) -> None:
    subtitles = [
        srt.Subtitle(
            index=segment.index,
            start=timedelta(seconds=segment.start),
            end=timedelta(seconds=segment.end),
            content=segment.text.strip(),
        )
        for segment in segments
        if segment.text.strip()
    ]
    path.write_text(srt.compose(subtitles), encoding="utf-8")


def bilingual_segments(
    source_segments: list[SubtitleSegment],
    translated_segments: list[SubtitleSegment],
    *,
    source_first: bool = True,
) -> list[SubtitleSegment]:
    if len(source_segments) != len(translated_segments):
        raise ValueError(
            f"subtitle count mismatch: {len(source_segments)} source, "
            f"{len(translated_segments)} translated"
        )

    segments: list[SubtitleSegment] = []
    for source, translated in zip(source_segments, translated_segments, strict=True):
        if source.index != translated.index:
            raise ValueError(
                f"subtitle index mismatch: source {source.index}, translated {translated.index}"
            )
        texts = [source.text.strip(), translated.text.strip()]
        if not source_first:
            texts.reverse()
        segments.append(
            SubtitleSegment(
                index=source.index,
                start=source.start,
                end=source.end,
                text="\n".join(text for text in texts if text),
            )
        )
    return segments
