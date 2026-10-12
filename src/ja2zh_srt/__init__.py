from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import click
from audio_transcribe import Transcriber, compute_type_for

from .subtitles import write_srt
from .translate import SakuraConfig, TranslationError, default_translated_path, translate_file

MP3_SUFFIX = ".mp3"
JA_SRT_SUFFIX = ".ja.srt"


@dataclass(frozen=True)
class TranscriptionJob:
    audio_path: Path
    output_path: Path


@dataclass(frozen=True)
class TranslationJob:
    input_path: Path
    output_path: Path


@dataclass(frozen=True)
class PipelineOptions:
    force: bool
    whisper_model: str
    device: str
    compute_type: str | None
    beam_size: int
    vad: bool
    batch_size: int
    temperature: float
    top_p: float
    llm_url: str
    api_key: str
    llm_model: str


class Ja2ZhSrtGroup(click.Group):
    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        if args and args[0] not in self.commands and not args[0].startswith("-"):
            args = ["run", *args]
        return super().parse_args(ctx, args)


@click.group(
    cls=Ja2ZhSrtGroup,
    no_args_is_help=True,
    help=(
        "Generate .ja.srt and .zh.srt subtitles from Japanese MP3 files.\n\n"
        "Use `ja2zh-srt <filename-or-directory> [-f]` to run the pipeline."
    ),
)
def main() -> None:
    pass


@main.command()
@click.option("--device", type=click.Choice(["cuda", "cpu"]), default="cuda")
@click.option(
    "--llm-url", envvar="JA2ZH_SRT_LLM_URL", default="http://127.0.0.1:4000/v1", show_default=True
)
@click.option("--api-key", envvar="JA2ZH_SRT_API_KEY", default="not-needed")
def check(device: str, llm_url: str, api_key: str) -> None:
    """Check transcription acceleration and the configured LLM endpoint."""
    import httpx

    try:
        print(f"transcription: device={device} compute={compute_type_for(device)}")
        response = httpx.get(
            llm_url.rstrip("/") + "/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10,
        )
        response.raise_for_status()
        print(f"LLM endpoint: {llm_url}; models={[m['id'] for m in response.json()['data']]}")
    except (RuntimeError, OSError, httpx.HTTPError) as error:
        raise click.ClickException(str(error)) from error


@main.command("run", hidden=True)
@click.argument("target", type=click.Path(exists=True, path_type=Path))
@click.option("-f", "--force", is_flag=True, help="Overwrite existing .ja.srt/.zh.srt files.")
@click.option("--whisper-model", default="large-v3", show_default=True)
@click.option("--device", type=click.Choice(["cuda", "cpu"]), default="cuda", show_default=True)
@click.option("--compute-type", default=None)
@click.option("--beam-size", default=5, show_default=True, type=click.IntRange(1))
@click.option("--vad/--no-vad", default=True, show_default=True)
@click.option("--batch-size", default=8, show_default=True, type=click.IntRange(1))
@click.option("--temperature", default=0.3, show_default=True, type=click.FloatRange(0.0))
@click.option("--top-p", default=0.8, show_default=True, type=click.FloatRange(0.0, 1.0))
@click.option(
    "--llm-url", envvar="JA2ZH_SRT_LLM_URL", default="http://127.0.0.1:4000/v1", show_default=True
)
@click.option(
    "--api-key",
    envvar="JA2ZH_SRT_API_KEY",
    default="not-needed",
    help="LLM API key; prefer JA2ZH_SRT_API_KEY environment variable.",
)
@click.option("--llm-model", envvar="JA2ZH_SRT_LLM_MODEL", default="sakura", show_default=True)
def run(target: Path, **kwargs) -> None:
    """Run the subtitle pipeline using the audio-transcribe SDK and an LLM API."""
    try:
        process_target(target, PipelineOptions(**kwargs))
    except (TranslationError, OSError, ValueError, RuntimeError) as exc:
        raise click.ClickException(str(exc)) from exc


def process_target(target: Path, options: PipelineOptions) -> None:
    if target.is_dir():
        process_directory(target, options)
        return

    if not target.is_file():
        raise click.ClickException(f"not a file or directory: {target}")

    if target.suffix.lower() == MP3_SUFFIX:
        process_mp3_file(target, options)
        return

    if is_ja_srt(target):
        process_ja_srt_file(target, options)
        return

    raise click.ClickException(f"expected an .mp3 file, .ja.srt file, or directory: {target}")


def process_mp3_file(audio_path: Path, options: PipelineOptions) -> None:
    ja_path = japanese_subtitle_path(audio_path)
    zh_path = chinese_subtitle_path(audio_path)

    if options.force or not ja_path.exists():
        run_transcriptions([TranscriptionJob(audio_path, ja_path)], options)
    else:
        print(f"skip existing {ja_path}")

    if not ja_path.exists():
        raise click.ClickException(f"missing Japanese subtitles after transcription: {ja_path}")

    if options.force or not zh_path.exists():
        run_translations([TranslationJob(ja_path, zh_path)], options)
    else:
        print(f"skip existing {zh_path}")


def process_ja_srt_file(subtitle_path: Path, options: PipelineOptions) -> None:
    zh_path = default_translated_path(subtitle_path)
    if options.force or not zh_path.exists():
        run_translations([TranslationJob(subtitle_path, zh_path)], options)
    else:
        print(f"skip existing {zh_path}")


def process_directory(directory: Path, options: PipelineOptions) -> None:
    audio_paths = sorted(path for path in directory.rglob("*.mp3") if path.is_file())
    if audio_paths:
        print(f"found {len(audio_paths)} mp3 file(s)")
    else:
        print(f"no mp3 files found under {directory}")

    transcription_jobs = [
        TranscriptionJob(audio_path, japanese_subtitle_path(audio_path))
        for audio_path in audio_paths
        if options.force or not japanese_subtitle_path(audio_path).exists()
    ]
    if not options.force:
        for audio_path in audio_paths:
            ja_path = japanese_subtitle_path(audio_path)
            if ja_path.exists():
                print(f"skip existing {ja_path}")
    if audio_paths:
        run_transcriptions(transcription_jobs, options)

    ja_paths = sorted(path for path in directory.rglob(f"*{JA_SRT_SUFFIX}") if path.is_file())
    if not ja_paths:
        print(f"no .ja.srt files found under {directory}")
        return

    translation_jobs = [
        TranslationJob(ja_path, default_translated_path(ja_path))
        for ja_path in ja_paths
        if options.force or not default_translated_path(ja_path).exists()
    ]
    if not options.force:
        for ja_path in ja_paths:
            zh_path = default_translated_path(ja_path)
            if zh_path.exists():
                print(f"skip existing {zh_path}")
    run_translations(translation_jobs, options)


def run_transcriptions(jobs: list[TranscriptionJob], options: PipelineOptions) -> None:
    if not jobs:
        print("transcription skipped; all .ja.srt files already exist")
        return

    print(f"transcribing {len(jobs)} file(s)")
    with Transcriber(
        model=options.whisper_model,
        device=options.device,
        compute_type=options.compute_type,
    ) as transcriber:
        for job in jobs:
            from .subtitles import SubtitleSegment

            result = transcriber.transcribe(
                job.audio_path,
                language="ja",
                beam_size=options.beam_size,
                vad_filter=options.vad,
                condition_on_previous_text=False,
            )
            segments = [
                SubtitleSegment(index=i, start=s.start, end=s.end, text=s.text)
                for i, s in enumerate(result.segments, 1)
            ]
            write_srt(segments, job.output_path)
            print(f"wrote {job.output_path}")


def run_translations(jobs: list[TranslationJob], options: PipelineOptions) -> None:
    if not jobs:
        print("translation skipped; all .zh.srt files already exist")
        return

    print(f"translating {len(jobs)} file(s)")
    config = SakuraConfig(
        base_url=options.llm_url,
        api_key=options.api_key,
        model=options.llm_model,
        temperature=options.temperature,
        top_p=options.top_p,
    )
    for job in jobs:
        translate_file(
            job.input_path, job.output_path, config=config, batch_size=options.batch_size
        )
        print(f"wrote {job.output_path}")


def japanese_subtitle_path(audio_path: Path) -> Path:
    return audio_path.with_suffix(JA_SRT_SUFFIX)


def chinese_subtitle_path(audio_path: Path) -> Path:
    return audio_path.with_suffix(".zh.srt")


def is_ja_srt(path: Path) -> bool:
    return path.name.lower().endswith(JA_SRT_SUFFIX)
