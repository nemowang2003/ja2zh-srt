from __future__ import annotations

import os
import socket
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import TextIO

from .config import ensure_cache_dir
from .models import DEFAULT_SAKURA_MODEL, sakura_model_path
from .translate import SakuraConfig, TranslationError


def default_sakura_model_path() -> Path:
    return sakura_model_path(DEFAULT_SAKURA_MODEL)


def iq4_xs_sakura_model_path() -> Path:
    return sakura_model_path("iq4-xs")


@dataclass(frozen=True)
class SakuraServerConfig:
    model_path: Path = field(default_factory=default_sakura_model_path)
    server_command: str = "llama-server"
    host: str = "127.0.0.1"
    port: int = 0
    alias: str = "sakura"
    api_key: str = "not-needed"
    ctx_size: int = 4096
    gpu_layers: str = "all"
    threads: int = 8
    threads_batch: int = 8
    llama_batch_size: int = 1024
    llama_ubatch_size: int = 256
    parallel: int = 1
    flash_attn: str = "on"
    cache_type_k: str = "q8_0"
    cache_type_v: str = "q8_0"
    fit: str = "on"
    fit_target_mib: int = 512
    startup_timeout: float = 180.0
    log_path: Path | None = None

    @classmethod
    def from_options(
        cls,
        *,
        model_path: Path | None = None,
        server_command: str = "llama-server",
        host: str = "127.0.0.1",
        port: int = 0,
        alias: str = "sakura",
        api_key: str = "not-needed",
        ctx_size: int = 4096,
        gpu_layers: str = "all",
        threads: int = 8,
        threads_batch: int = 8,
        llama_batch_size: int = 1024,
        llama_ubatch_size: int = 256,
        parallel: int = 1,
        flash_attn: str = "on",
        cache_type_k: str = "q8_0",
        cache_type_v: str = "q8_0",
        fit: str = "on",
        fit_target_mib: int = 512,
        startup_timeout: float = 180.0,
        log_path: Path | None = None,
    ) -> SakuraServerConfig:
        env_model_path = os.environ.get("SAKURA_MODEL_PATH")
        return cls(
            model_path=Path(env_model_path).expanduser()
            if env_model_path
            else model_path or default_sakura_model_path(),
            server_command=server_command,
            host=os.environ.get("SAKURA_HOST", host),
            port=int(os.environ.get("SAKURA_PORT", port)),
            alias=os.environ.get("SAKURA_MODEL", alias),
            api_key=os.environ.get("SAKURA_API_KEY", api_key),
            ctx_size=int(os.environ.get("SAKURA_CTX_SIZE", ctx_size)),
            gpu_layers=os.environ.get("SAKURA_GPU_LAYERS", gpu_layers),
            threads=int(os.environ.get("SAKURA_THREADS", threads)),
            threads_batch=threads_batch,
            llama_batch_size=llama_batch_size,
            llama_ubatch_size=llama_ubatch_size,
            parallel=parallel,
            flash_attn=flash_attn,
            cache_type_k=cache_type_k,
            cache_type_v=cache_type_v,
            fit=fit,
            fit_target_mib=fit_target_mib,
            startup_timeout=startup_timeout,
            log_path=log_path,
        )


class ManagedSakuraServer:
    def __init__(
        self,
        config: SakuraServerConfig,
        *,
        temperature: float = 0.3,
        top_p: float = 0.8,
    ) -> None:
        self.config = config
        self.temperature = temperature
        self.top_p = top_p
        self._process: subprocess.Popen[str] | None = None
        self._log_file: TextIO | None = None
        self._port = config.port

    def __enter__(self) -> SakuraConfig:
        model_path = self.config.model_path.expanduser()
        if not model_path.is_file():
            fallback_path = iq4_xs_sakura_model_path()
            fallback_message = (
                f"; pass --sakura-model-path {fallback_path} to use the existing IQ4_XS model"
                if fallback_path.is_file()
                else ""
            )
            raise TranslationError(
                f"missing Sakura model: {model_path}; "
                "run: ja2zh-srt download sakura"
                f"{fallback_message}"
            )

        if self._port == 0:
            self._port = _find_free_port(self.config.host)

        log_path = self.config.log_path or ensure_cache_dir() / "logs" / "sakura-server.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_file = log_path.open("w", encoding="utf-8")

        command = self._build_command(model_path)
        try:
            self._process = subprocess.Popen(
                command,
                stdout=self._log_file,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
        except FileNotFoundError as exc:
            self._close_log()
            raise TranslationError(
                f"could not find Sakura server command: {self.config.server_command}"
            ) from exc

        base_url = f"http://{_connect_host(self.config.host)}:{self._port}/v1"
        try:
            self._wait_until_ready(base_url, log_path)
        except BaseException:
            self.stop()
            raise
        return SakuraConfig(
            base_url=base_url,
            api_key=self.config.api_key,
            model=self.config.alias,
            temperature=self.temperature,
            top_p=self.top_p,
        )

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self.stop()

    def stop(self) -> None:
        process = self._process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=30)
        self._process = None
        self._close_log()

    def _build_command(self, model_path: Path) -> list[str]:
        config = self.config
        return [
            config.server_command,
            "--model",
            str(model_path),
            "--alias",
            config.alias,
            "--host",
            config.host,
            "--port",
            str(self._port),
            "--ctx-size",
            str(config.ctx_size),
            "--n-gpu-layers",
            config.gpu_layers,
            "--threads",
            str(config.threads),
            "--threads-batch",
            str(config.threads_batch),
            "--batch-size",
            str(config.llama_batch_size),
            "--ubatch-size",
            str(config.llama_ubatch_size),
            "--parallel",
            str(config.parallel),
            "--flash-attn",
            config.flash_attn,
            "--cache-type-k",
            config.cache_type_k,
            "--cache-type-v",
            config.cache_type_v,
            "--fit",
            config.fit,
            "--fit-target",
            str(config.fit_target_mib),
            "--cont-batching",
            "--cache-prompt",
            "--no-ui",
        ]

    def _wait_until_ready(self, base_url: str, log_path: Path) -> None:
        assert self._process is not None
        deadline = time.monotonic() + self.config.startup_timeout
        url = f"{base_url}/models"
        request = urllib.request.Request(
            url,
            headers={"Authorization": f"Bearer {self.config.api_key}"},
        )

        while time.monotonic() < deadline:
            return_code = self._process.poll()
            if return_code is not None:
                raise TranslationError(
                    f"Sakura server exited with code {return_code}; log: {log_path}\n"
                    f"{_tail(log_path)}"
                )
            try:
                with urllib.request.urlopen(request, timeout=2) as response:
                    if 200 <= response.status < 500:
                        return
            except (urllib.error.URLError, TimeoutError):
                time.sleep(0.5)

        raise TranslationError(
            f"Sakura server did not become ready within "
            f"{self.config.startup_timeout:.0f}s; log: {log_path}\n{_tail(log_path)}"
        )

    def _close_log(self) -> None:
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None


def _find_free_port(host: str) -> int:
    bind_host = _connect_host(host)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((bind_host, 0))
        return int(sock.getsockname()[1])


def _connect_host(host: str) -> str:
    if host in {"0.0.0.0", "::"}:
        return "127.0.0.1"
    return host


def _tail(path: Path, *, limit: int = 4000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-limit:]
