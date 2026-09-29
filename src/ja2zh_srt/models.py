from __future__ import annotations

import shutil
from importlib import metadata
from pathlib import Path

from faster_whisper.utils import download_model
from huggingface_hub import hf_hub_download

from .config import ensure_cache_dir

SAKURA_REPO_ID = "SakuraLLM/Sakura-GalTransl-7B-v3.7"
SAKURA_MODEL_FILES = {
    "iq4-xs": "Sakura-Galtransl-7B-v3.7-IQ4_XS.gguf",
    "q6-k": "Sakura-Galtransl-7B-v3.7.gguf",
}
DEFAULT_SAKURA_MODEL = "q6-k"
DEFAULT_SAKURA_VARIANT = DEFAULT_SAKURA_MODEL
DEFAULT_WHISPER_MODEL = "large-v3"


def faster_whisper_cache_dir() -> Path:
    return ensure_cache_dir() / "faster-whisper"


def sakura_cache_dir() -> Path:
    return ensure_cache_dir() / "sakura"


def sakura_model_file(model_name: str = DEFAULT_SAKURA_MODEL) -> str:
    normalized = model_name.lower().replace("_", "-")
    return SAKURA_MODEL_FILES.get(normalized, model_name)


def sakura_model_path(model_name: str = DEFAULT_SAKURA_MODEL) -> Path:
    return sakura_cache_dir() / sakura_model_file(model_name)


def download_sakura_model(
    model_name: str = DEFAULT_SAKURA_MODEL,
    *,
    force: bool = False,
) -> Path:
    local_dir = sakura_cache_dir()
    local_dir.mkdir(parents=True, exist_ok=True)
    return Path(
        hf_hub_download(
            repo_id=SAKURA_REPO_ID,
            filename=sakura_model_file(model_name),
            local_dir=local_dir,
            force_download=force,
        )
    )


def download_whisper_model(
    model_name: str = DEFAULT_WHISPER_MODEL,
    *,
    force: bool = False,
) -> Path:
    configured_path = Path(model_name).expanduser()
    if configured_path.exists():
        return configured_path

    cache_root = faster_whisper_cache_dir()
    cache_root.mkdir(parents=True, exist_ok=True)
    model_cache_path = whisper_model_cache_path(model_name)
    if force and model_cache_path.exists() and _is_relative_to(model_cache_path, cache_root):
        shutil.rmtree(model_cache_path)

    return Path(
        download_model(
            model_name,
            cache_dir=str(cache_root),
            local_files_only=False,
        )
    )


def print_status(
    *,
    whisper_model: str = DEFAULT_WHISPER_MODEL,
    sakura_model: str = DEFAULT_SAKURA_MODEL,
) -> None:
    print(f"cache_dir: {ensure_cache_dir()}")
    _print_executable("llama-server", "required for running the Sakura GGUF model")
    _print_executable(
        "ffmpeg",
        "optional; useful for audio conversion/debugging, but ja2zh-srt does not call it directly",
        required=False,
    )
    _print_package("faster-whisper")
    _print_whisper_model(whisper_model)
    _print_sakura_model(sakura_model)


def whisper_model_cache_path(model_name: str) -> Path:
    configured_path = Path(model_name).expanduser()
    if configured_path.exists():
        return configured_path
    repo_id = model_name if "/" in model_name else f"Systran/faster-whisper-{model_name}"
    return faster_whisper_cache_dir() / f"models--{repo_id.replace('/', '--')}"


def _print_executable(name: str, detail: str, *, required: bool = True) -> None:
    path = shutil.which(name)
    if path:
        print(f"ok: {name} path={path} ({detail})")
    else:
        state = "missing" if required else "optional missing"
        print(f"{state}: {name} ({detail})")


def _print_package(package_name: str) -> None:
    try:
        version = metadata.version(package_name)
    except metadata.PackageNotFoundError:
        print(f"missing: {package_name} package")
        return
    print(f"ok: {package_name} package version={version}")


def _print_whisper_model(model_name: str) -> None:
    path = whisper_model_cache_path(model_name)
    if _whisper_model_available(path):
        print(f"ok: faster-whisper:{model_name} path={path}")
    else:
        print(f"missing: faster-whisper:{model_name} expected={path}")


def _print_sakura_model(model_name: str) -> None:
    path = sakura_model_path(model_name)
    if path.is_file():
        print(f"ok: sakura:{model_name} path={path}")
    else:
        print(f"missing: sakura:{model_name} expected={path}")


def _whisper_model_available(path: Path) -> bool:
    if not path.exists():
        return False
    if (path / "model.bin").is_file():
        return True
    snapshots = path / "snapshots"
    return snapshots.is_dir() and any(snapshot.is_dir() for snapshot in snapshots.iterdir())


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True
