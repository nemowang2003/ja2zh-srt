from __future__ import annotations

import re
import shutil
import subprocess
from importlib import metadata
from pathlib import Path

import ctranslate2
from faster_whisper.utils import download_model
from huggingface_hub import hf_hub_download

from .config import ensure_cache_dir

SAKURA_REPO_ID = "SakuraLLM/Sakura-GalTransl-7B-v3.7"
SAKURA_MODEL_FILES = {
    "iq4-xs": "Sakura-Galtransl-7B-v3.7-IQ4_XS.gguf",
    "q6-k": "Sakura-Galtransl-7B-v3.7.gguf",
}
DEFAULT_SAKURA_MODEL = "iq4-xs"
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
) -> bool:
    print(f"cache_dir: {ensure_cache_dir()}")
    server_ready = _print_llama_server()
    _print_executable(
        "ffmpeg",
        "optional; useful for audio conversion/debugging, but ja2zh-srt does not call it directly",
        required=False,
    )
    package_ready = _print_package("faster-whisper")
    cuda_ready = _print_cuda()
    whisper_ready = _print_whisper_model(whisper_model)
    sakura_ready = _print_sakura_model(sakura_model)
    return all((server_ready, package_ready, cuda_ready, whisper_ready, sakura_ready))


def whisper_model_cache_path(model_name: str) -> Path:
    configured_path = Path(model_name).expanduser()
    if configured_path.exists():
        return configured_path
    repo_id = model_name if "/" in model_name else f"Systran/faster-whisper-{model_name}"
    return faster_whisper_cache_dir() / f"models--{repo_id.replace('/', '--')}"


def _print_executable(name: str, detail: str, *, required: bool = True) -> bool:
    path = shutil.which(name)
    if path:
        print(f"ok: {name} path={path} ({detail})")
        return True
    else:
        state = "missing" if required else "optional missing"
        print(f"{state}: {name} ({detail})")
        return not required


def _print_llama_server() -> bool:
    path = shutil.which("llama-server")
    if path is None:
        print("missing: llama-server executable")
        return False
    try:
        result = subprocess.run(
            [path, "--list-devices"], capture_output=True, text=True, timeout=15, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"missing: llama-server CUDA device check failed ({exc})")
        return False
    devices = re.findall(r"\bCUDA\d+: [^\r\n]+", result.stdout + result.stderr)
    if result.returncode or not devices:
        print(f"missing: llama-server cannot see a CUDA device (path={path})")
        return False
    print(f"ok: llama-server path={path} device={devices[0]}")
    return True


def _print_package(package_name: str) -> bool:
    try:
        version = metadata.version(package_name)
    except metadata.PackageNotFoundError:
        print(f"missing: {package_name} package")
        return False
    print(f"ok: {package_name} package version={version}")
    return True


def _print_cuda() -> bool:
    try:
        devices = ctranslate2.get_cuda_device_count()
        if devices:
            compute_types = ctranslate2.get_supported_compute_types("cuda")
            if "float16" in compute_types:
                print(f"ok: CUDA devices={devices} faster-whisper compute=float16")
                return True
            print(f"missing: CUDA float16 support (available: {sorted(compute_types)})")
        else:
            print("missing: CUDA device not visible to faster-whisper")
    except RuntimeError as exc:
        print(f"missing: CUDA runtime for faster-whisper ({exc})")
    return False


def _print_whisper_model(model_name: str) -> bool:
    path = whisper_model_cache_path(model_name)
    if _whisper_model_available(path):
        print(f"ok: faster-whisper:{model_name} path={path}")
        return True
    else:
        print(f"missing: faster-whisper:{model_name} expected={path}")
        return False


def _print_sakura_model(model_name: str) -> bool:
    path = sakura_model_path(model_name)
    if path.is_file():
        print(f"ok: sakura:{model_name} path={path}")
        return True
    else:
        print(f"missing: sakura:{model_name} expected={path}")
        return False


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
