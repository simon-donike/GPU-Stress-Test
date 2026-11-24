"""Typed stop conditions and a small, dependency-free child failure channel."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


class ThermalGuardError(RuntimeError):
    """The sampled temperature guard failed or its telemetry disappeared."""


class NonFiniteLossError(RuntimeError):
    """The training loss is missing or non-finite."""


class InsufficientMemoryError(RuntimeError):
    """Model state cannot fit within the preflight memory budget."""


def failure_kind(exception: BaseException) -> str:
    if isinstance(exception, KeyboardInterrupt):
        return "interrupted"
    if isinstance(exception, ThermalGuardError):
        return "thermal"
    if isinstance(exception, NonFiniteLossError):
        return "nonfinite"
    if isinstance(exception, InsufficientMemoryError):
        return "memory_preflight"
    torch = sys.modules.get("torch")
    oom_type = getattr(torch, "OutOfMemoryError", None)
    if oom_type is not None and isinstance(exception, oom_type):
        return "oom"
    return "internal_error"


def record_failure(directory: Path | None, exception: BaseException, rank: int | None = None) -> str:
    """Record a caught error per rank; never infer OOM from an exit signal."""
    kind = failure_kind(exception)
    if directory is not None:
        try:
            rank = int(rank if rank is not None else os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
            payload = {"rank": rank, "failure_kind": kind, "exception_type": type(exception).__name__, "message": str(exception)}
            directory.mkdir(parents=True, exist_ok=True)
            # Keep the typed cause even when Lightning teardown subsequently
            # raises a generic error in the same rank/process.
            path = directory / f"failure_rank{rank}_{os.getpid()}_{kind}.json"
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            temporary.replace(path)
        except (OSError, ValueError, TypeError):
            # Never hide the original training error because failure reporting failed.
            pass
    return kind
