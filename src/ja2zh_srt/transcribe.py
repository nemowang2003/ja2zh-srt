from __future__ import annotations

import gc
from pathlib import Path

from faster_whisper import WhisperModel

from .models import faster_whisper_cache_dir
from .subtitles import SubtitleSegment


class Transcriber:
    def __init__(
        self,
        *,
        model_name: str = "large-v3",
        device: str = "cuda",
        compute_type: str = "float16",
        language: str = "ja",
        beam_size: int = 5,
        vad_filter: bool = True,
    ) -> None:
        self.language = language
        self.beam_size = beam_size
        self.vad_filter = vad_filter
        self._model: WhisperModel | None = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
            download_root=str(faster_whisper_cache_dir()),
        )

    def __enter__(self) -> Transcriber:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def transcribe(self, audio_path: Path) -> list[SubtitleSegment]:
        if self._model is None:
            raise RuntimeError("transcriber has been closed")
        segments, _info = self._model.transcribe(
            str(audio_path),
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=self.vad_filter,
            condition_on_previous_text=False,
        )
        return [
            SubtitleSegment(
                index=index,
                start=segment.start,
                end=segment.end,
                text=segment.text.strip(),
            )
            for index, segment in enumerate(segments, start=1)
        ]

    def close(self) -> None:
        self._model = None
        release_transcription_memory()


def release_transcription_memory() -> None:
    gc.collect()


def transcribe_audio(
    audio_path: Path,
    *,
    model_name: str = "large-v3",
    device: str = "cuda",
    compute_type: str = "float16",
    language: str = "ja",
    beam_size: int = 5,
    vad_filter: bool = True,
) -> list[SubtitleSegment]:
    with Transcriber(
        model_name=model_name,
        device=device,
        compute_type=compute_type,
        language=language,
        beam_size=beam_size,
        vad_filter=vad_filter,
    ) as transcriber:
        return transcriber.transcribe(audio_path)
