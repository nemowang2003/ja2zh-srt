# ja2zh-srt

`ja2zh-srt` 是一个 CUDA-only 命令行工具，用于把日语 MP3 音频生成日语
SRT 字幕，再通过 Sakura LLM 翻译成中文字幕。

当前支持：

- `*.mp3` 转写为 `*.ja.srt`
- `*.ja.srt` 翻译为 `*.zh.srt`
- 递归处理目录下的 MP3 和日语 SRT
- 自动检查和下载 faster-whisper / Sakura 模型
- Nix 和 Pixi 两种 CUDA 部署入口

运行时模型缓存目录可通过 `JA2ZH_SRT_CACHE_DIR` 设置，默认是：

```text
~/.cache/ja2zh-srt
```

## CUDA 要求

这个项目只面向 NVIDIA CUDA 环境，不维护 CPU、Metal、ROCm 或其他后端。

必须有：

- Python 3.12
- NVIDIA 驱动和 CUDA 12 runtime
- CUDA 可用的 faster-whisper / CTranslate2
- CUDA 版 `llama-server`

`ffmpeg` 会被 `check` 检查，也会被 Pixi/Nix 环境安装。当前 Python 代码不直接调用外部
`ffmpeg`；它主要用于把其他音频格式预处理成 MP3，或排查音频文件问题。MP3 解码本身走
faster-whisper/PyAV 依赖路径。

## 安装

直接使用本机环境：

```bash
uv sync
```

使用 Pixi 安装 CUDA 版 `llama.cpp`、`llama-server`、`ffmpeg`、Python 3.12 和 `uv`：

```bash
pixi install
pixi shell
uv sync
```

使用 Nix：

```bash
nix develop --accept-flake-config
```

或直接运行打包后的命令：

```bash
nix run . -- check
```

Nix flake 使用 `uv2nix` 从 `uv.lock` 构建 Python 环境，并把 `ffmpeg` 和 CUDA 版
`llama-server` 放到运行时 PATH。当前 flake 只暴露 `x86_64-linux`。

## 命令

检查运行环境和模型：

```bash
ja2zh-srt check
```

下载默认 faster-whisper 模型：

```bash
ja2zh-srt download whisper
```

下载指定 faster-whisper 模型，强制重新下载：

```bash
ja2zh-srt download whisper large-v3 -f
```

下载默认 Sakura GGUF 模型。默认是 `q6-k`：

```bash
ja2zh-srt download sakura
```

下载较小的 Sakura IQ4_XS：

```bash
ja2zh-srt download sakura iq4-xs
```

处理单个 MP3：

```bash
ja2zh-srt input.mp3
```

规则：

- 如果 `input.ja.srt` 不存在，先转写生成它。
- 如果 `input.ja.srt` 存在但 `input.zh.srt` 不存在，翻译生成中文字幕。
- 已存在的输出默认跳过；传 `-f` 会覆盖重新生成。

处理单个日语 SRT：

```bash
ja2zh-srt input.ja.srt
```

如果 `input.zh.srt` 不存在，则翻译；传 `-f` 会覆盖。

递归处理目录：

```bash
ja2zh-srt /path/to/directory
```

规则：

- 递归查找目录下所有 `*.mp3`。
- 对缺失对应 `*.ja.srt` 的 MP3 执行转写。
- 然后递归查找目录下所有 `*.ja.srt`。
- 对缺失对应 `*.zh.srt` 的日语 SRT 执行翻译。
- 传 `-f` 会覆盖所有对应输出。

## Pipeline

```text
mp3 file
  -> faster-whisper on CUDA
  -> Japanese SRT
  -> managed Sakura/llama.cpp CUDA OpenAI-compatible API
  -> Chinese SRT
```

翻译阶段只替换字幕文本，不让 LLM 生成或修改时间轴。Python 会把字幕文本整理成编号纯文本块发给
Sakura，并校验返回的编号集合；输出字幕会保留原始 cue 数量、顺序和时间戳。

## 验证

```bash
uv run ja2zh-srt --help
uv run ja2zh-srt check
uv run ruff check .
python -m compileall -q src
```

Nix 求值/干跑：

```bash
nix flake show --allow-import-from-derivation
nix build .#checks.x86_64-linux.help --dry-run --accept-flake-config --allow-import-from-derivation
```
