from __future__ import annotations

from pathlib import Path
import logging
import os
from typing import Optional

import torch

_OPS = None
_OPS_LOAD_ERROR: Optional[Exception] = None
_OPS_LOGGED = False

_LOG = logging.getLogger("gpt_oss.mojo.mxfp4")


def _truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _should_use_mojo() -> bool:
    mode = os.environ.get("GPT_OSS_MOJO_MXFP4", "auto").strip().lower()
    if mode == "auto":
        return True
    return _truthy(mode)


def _should_strict() -> bool:
    return _truthy(os.environ.get("GPT_OSS_MOJO_MXFP4_STRICT", "0"))


def _log_once(message: str, level: str = "info") -> None:
    global _OPS_LOGGED
    if _OPS_LOGGED:
        return
    _OPS_LOGGED = True
    getattr(_LOG, level, _LOG.info)(message)


def _resolve_ops_path() -> Path:
    override = os.environ.get("GPT_OSS_MOJO_MXFP4_PATH")
    if override:
        return Path(override)
    ops_dir = Path(__file__).parent / "operations"
    if (ops_dir / "__init__.mojo").exists():
        return ops_dir
    for candidate in (ops_dir / "mxfp4.mojopkg", ops_dir / "mxfp4.mojo"):
        if candidate.exists():
            return candidate
    return ops_dir


def _load_ops():
    global _OPS, _OPS_LOAD_ERROR
    if _OPS is not None or _OPS_LOAD_ERROR is not None:
        return _OPS
    try:
        from max.torch import CustomOpLibrary
    except Exception as exc:
        _OPS_LOAD_ERROR = exc
        return None
    ops_path = _resolve_ops_path()
    try:
        _OPS = CustomOpLibrary(ops_path)
    except Exception as exc:
        _OPS_LOAD_ERROR = exc
        return None
    _log_once(f"Mojo MXFP4 op loaded from {ops_path}", "warning")
    return _OPS


def try_mxfp4_unpack(
    out: torch.Tensor,
    blocks: torch.Tensor,
    scales: torch.Tensor,
) -> bool:
    if not _should_use_mojo():
        return False
    ops = _load_ops()
    if ops is None:
        if _should_strict():
            raise RuntimeError(f"Mojo MXFP4 op failed to load: {_OPS_LOAD_ERROR!r}")
        _log_once("Mojo MXFP4 op unavailable; falling back to torch", "warning")
        return False
    try:
        ops.mxfp4_unpack(out, blocks, scales)
    except Exception:  # pragma: no cover - runtime fallback
        if _should_strict():
            raise
        _log_once("Mojo MXFP4 op failed at runtime; falling back to torch", "warning")
        return False
    return True
