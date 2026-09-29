from __future__ import annotations

import os
from pathlib import Path

import platformdirs

PROJECT_NAME = "ja2zh-srt"
CACHE_DIR_ENV_VAR = "JA2ZH_SRT_CACHE_DIR"


def cache_dir() -> Path:
    configured = os.environ.get(CACHE_DIR_ENV_VAR)
    if configured:
        return Path(os.path.expandvars(configured)).expanduser()
    return Path(platformdirs.user_cache_dir(PROJECT_NAME))


def ensure_cache_dir() -> Path:
    path = cache_dir()
    path.mkdir(parents=True, exist_ok=True)
    return path
