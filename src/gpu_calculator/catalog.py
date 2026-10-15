"""Справочник моделей и GPU из data/*.json."""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

from .formulas import GPU, Model

DATA_DIR = Path(__file__).parent / "data"


def _load(path: Path, cls: type) -> dict:
    entries = json.loads(path.read_text(encoding="utf-8"))
    return {e["name"]: cls(**e) for e in entries}


@cache
def models() -> dict[str, Model]:
    return _load(DATA_DIR / "models.json", Model)


@cache
def gpus() -> dict[str, GPU]:
    return _load(DATA_DIR / "gpus.json", GPU)


def model(name: str) -> Model:
    try:
        return models()[name]
    except KeyError:
        raise KeyError(f"Неизвестная модель: {name}. Есть: {', '.join(models())}") from None


def gpu(name: str) -> GPU:
    try:
        return gpus()[name]
    except KeyError:
        raise KeyError(f"Неизвестная GPU: {name}. Есть: {', '.join(gpus())}") from None
