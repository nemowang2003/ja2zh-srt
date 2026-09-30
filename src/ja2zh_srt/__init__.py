from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import click

from . import models as model_store
from .sakura import ManagedSakuraServer, SakuraServerConfig
from .subtitles import write_srt
from .transcribe import Transcriber, release_transcription_memory
from .translate import TranslationError, default_translated_path, translate_file

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
    compute_type: str
    beam_size: int
    vad: bool
    batch_size: int
    temperature: float
    top_p: float
    sakura_model: str
    sakura_model_path: Path | None
    sakura_server_command: str
    sakura_host: str
    sakura_port: int
    sakura_startup_timeout: float


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
@click.option("--whisper-model", default=model_store.DEFAULT_WHISPER_MODEL, show_default=True)
@click.option("--sakura-model", default=model_store.DEFAULT_SAKURA_MODEL, show_default=True)
def check(whisper_model: str, sakura_model: str) -> None:
    """Check runtime tools and cached models."""
    if not model_store.print_status(whisper_model=whisper_model, sakura_model=sakura_model):
        sys.stdout.flush()
        raise click.ClickException("runtime is not ready; resolve the missing items above")


@main.group()
def download() -> None:
    """Download runtime models."""


@download.command("whisper")
@click.argument("model_name", required=False, default=model_store.DEFAULT_WHISPER_MODEL)
@click.option("-f", "--force", is_flag=True, help="Remove the local cache and download again.")
def download_whisper(model_name: str, force: bool) -> None:
    """Download a faster-whisper model."""
    path = model_store.download_whisper_model(model_name, force=force)
    print(f"wrote {path}")


@download.command("sakura")
@click.argument("model_name", required=False, default=model_store.DEFAULT_SAKURA_MODEL)
@click.option("-f", "--force", is_flag=True, help="Download again even if the file is cached.")
def download_sakura(model_name: str, force: bool) -> None:
    """Download a Sakura GGUF model."""
    path = model_store.download_sakura_model(model_name, force=force)
    print(f"wrote {path}")


@main.command("run", hidden=True)
@click.argument("target", type=click.Path(exists=True, path_type=Path))
@click.option("-f", "--force", is_flag=True, help="Overwrite existing .ja.srt/.zh.srt files.")
@click.option("--whisper-model", default=model_store.DEFAULT_WHISPER_MODEL, show_default=True)
@click.option("--device", default="cuda", show_default=True)
@click.option("--compute-type", default="float16", show_default=True)
@click.option("--beam-size", default=5, show_default=True, type=click.IntRange(1))
@click.option("--vad/--no-vad", default=True, show_default=True)
@click.option("--batch-size", default=8, show_default=True, type=click.IntRange(1))
@click.option("--temperature", default=0.3, show_default=True, type=click.FloatRange(0.0))
@click.option("--top-p", default=0.8, show_default=True, type=click.FloatRange(0.0, 1.0))
@click.option("--sakura-model", default=model_store.DEFAULT_SAKURA_MODEL, show_default=True)
@click.option(
    "--sakura-model-path",
    type=click.Path(file_okay=True, dir_okay=False, path_type=Path),
    default=None,
)
@click.option("--sakura-server-command", default="llama-server", show_default=True)
@click.option("--sakura-host", default="127.0.0.1", show_default=True)
@click.option("--sakura-port", default=0, show_default=True, type=click.IntRange(0, 65535))
@click.option(
    "--sakura-startup-timeout", default=180.0, show_default=True, type=click.FloatRange(1.0)
)
def run(
    target: Path,
    force: bool,
    whisper_model: str,
    device: str,
    compute_type: str,
    beam_size: int,
    vad: bool,
    batch_size: int,
    temperature: float,
    top_p: float,
    sakura_model: str,
    sakura_model_path: Path | None,
    sakura_server_command: str,
    sakura_host: str,
    sakura_port: int,
    sakura_startup_timeout: float,
) -> None:
    """Run the default subtitle pipeline."""
    options = PipelineOptions(
        force=force,
        whisper_model=whisper_model,
        device=device,
        compute_type=compute_type,
        beam_size=beam_size,
        vad=vad,
        batch_size=batch_size,
        temperature=temperature,
        top_p=top_p,
        sakura_model=sakura_model,
        sakura_model_path=sakura_model_path,
        sakura_server_command=sakura_server_command,
        sakura_host=sakura_host,
        sakura_port=sakura_port,
        sakura_startup_timeout=sakura_startup_timeout,
    )
    try:
        process_target(target, options)
    except TranslationError as exc:
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
        model_name=options.whisper_model,
        device=options.device,
        compute_type=options.compute_type,
        beam_size=options.beam_size,
        vad_filter=options.vad,
    ) as transcriber:
        for job in jobs:
            segments = transcriber.transcribe(job.audio_path)
            write_srt(segments, job.output_path)
            print(f"wrote {job.output_path}")
    release_transcription_memory()


def run_translations(jobs: list[TranslationJob], options: PipelineOptions) -> None:
    if not jobs:
        print("translation skipped; all .zh.srt files already exist")
        return

    print(f"translating {len(jobs)} file(s)")
    server_config = SakuraServerConfig.from_options(
        model_path=options.sakura_model_path or model_store.sakura_model_path(options.sakura_model),
        server_command=options.sakura_server_command,
        host=options.sakura_host,
        port=options.sakura_port,
        startup_timeout=options.sakura_startup_timeout,
    )
    with ManagedSakuraServer(
        server_config,
        temperature=options.temperature,
        top_p=options.top_p,
    ) as config:
        for job in jobs:
            translate_file(
                job.input_path,
                job.output_path,
                config=config,
                batch_size=options.batch_size,
            )
            print(f"wrote {job.output_path}")


def japanese_subtitle_path(audio_path: Path) -> Path:
    return audio_path.with_suffix(JA_SRT_SUFFIX)


def chinese_subtitle_path(audio_path: Path) -> Path:
    return audio_path.with_suffix(".zh.srt")


def is_ja_srt(path: Path) -> bool:
    return path.name.lower().endswith(JA_SRT_SUFFIX)
