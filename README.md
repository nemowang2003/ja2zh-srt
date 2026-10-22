# ja2zh-srt

日语音频 → 日语 SRT → 中文 SRT。转写通过 `audio-transcribe` Python SDK，翻译通过
兼容 OpenAI 的 LLM API。工具负责字幕流程，不下载模型、不启动 LLM 服务。

```sh
ja2zh-srt input.mp3
ja2zh-srt input.ja.srt
ja2zh-srt /path/to/directory
ja2zh-srt input.mp3 --llm-url http://127.0.0.1:4000/v1 --llm-model sakura
ja2zh-srt check
```

已有字幕默认跳过，`-f` 覆盖；目录先批量转写，再翻译。转写默认 CUDA／float16，
缺少 CUDA 就报错，只有 `--device cpu` 明确允许 CPU／int8。Whisper 默认 `large-v3`，
模型检查、下载、导入由 `audio-transcribe models` 管理，两者共享缓存及模型别名。

LLM 配置：`JA2ZH_SRT_LLM_URL`（默认 `http://127.0.0.1:4000/v1`）、
`JA2ZH_SRT_API_KEY`、`JA2ZH_SRT_LLM_MODEL`（默认 `sakura`）。也可以传
`--llm-url`、`--api-key`、`--llm-model`，密钥建议用环境变量。
请求超时为一小时，允许本地服务排队和冷启动；不自动重试整个请求。
若服务返回 `x-local-llm-warning`，CLI 显示可忽略的 RuntimeWarning，继续翻译。

翻译按编号校验输出，保留 cue 编号、数量和时间轴。响应不符合编号约定时拆小批次
重试，单条仍不合法就报错。日语转写使用 VAD、beam size 5，关闭 previous-text 条件。
仅翻译 SRT 时不会加载 Whisper 或检查本地 CUDA。

```sh
uv sync
uv run ja2zh-srt --help
nix run .#cuda -- input.mp3
nix flake check
nix build .#cuda
```

Python 依赖中直接声明 `audio-transcribe`，uv 使用其公开 Git tag。Nix flake 通过
`audio-transcribe` input 按相同 Python 版本构建 SDK；提供普通 CPU 包及 x86_64 Linux
CUDA 包。系统 CUDA/WSL 运行库由 Nix 包装器提供，其他加速后端暂未实现。
本项目不再需要 CUDA llama-server、Pixi 或外部 ffmpeg。

## Locked Nix builds

Nix builds resolve the project's `uv.lock` with uv2nix. Python dependencies,
including optional extras, come from that lock; nixpkgs supplies the interpreter,
system libraries and llama.cpp where applicable. The `build` dependency group
locks `uv_build` in `uv.lock`. Nix builds use `--force-pep517` and explicitly
provide that locked backend instead of uv's bundled version. Other build
backends come from the `pyproject-build-systems` flake input.

All three build inputs (`pyproject-nix`, `uv2nix`, `pyproject-build-systems`) may
follow a consuming flake's shared inputs. The standalone flake pins them itself.
`lib.mkPythonSet { pkgs; python; }` exposes the locked Python package scope for
SDK consumers; `lib.mkPackage { pkgs; python; cudaSupport; }` ships the CLI with
its isolated environment. The CLI's `python-env` passthru exposes that environment
for installation and runtime checks. Development shells supply Python and uv;
use `uv sync` / `uv run` for development, and `nix build` for deployment.

## Development checks

Ruff, ty and pre-commit are dev dependencies pinned in `uv.lock`. Install the
Git hook once per clone:

```sh
uv sync --locked
uv run --locked pre-commit install
uv run --locked pre-commit run --all-files
```

The hooks use the locked tools through uv: lock consistency, Ruff lint fixes,
Ruff formatting, and a project-wide ty check. Dev dependencies are excluded from
the deployed application environments.
