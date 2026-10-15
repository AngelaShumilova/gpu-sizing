"""Подбор числа GPU: сколько нужно и как быстро будет работать.

Несколько GPU считаются одним пулом: их память, пропускная способность и вычисления складываются.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .formulas import GPU, Model, Workload, kv_gb, memory_gb, prefill_seconds, tpot_seconds, weights_gb

DEFAULT_OVERHEAD = 1.3  # запас на фреймворк, CUDA-контекст и фрагментацию


@dataclass(frozen=True)
class Result:
    gpus: int
    weights_gb: float
    kv_cache_gb: float
    memory_gb: float  # (веса + KV-кэш) × запас
    memory_per_gpu_gb: float
    fits: bool
    ttft_s: float
    tpot_ms: float
    e2e_s: float


def gpus_for_memory(model: Model, gpu: GPU, workload: Workload, overhead: float = DEFAULT_OVERHEAD) -> int:
    """Сколько GPU нужно, чтобы всё поместилось: всего памяти / память одной GPU, вверх."""
    return math.ceil(memory_gb(model, workload, overhead) / gpu.memory_gb)


def evaluate(
    model: Model,
    gpu: GPU,
    workload: Workload,
    gpus: int,
    overhead: float = DEFAULT_OVERHEAD,
) -> Result:
    """Считает память и скорость для заданного числа GPU.

    TTFT = prefill: первый токен выходит из prefill.
    E2E = TTFT + (сгенерированных токенов − 1) × TPOT.
    """
    if gpus < 1:
        raise ValueError("gpus должно быть >= 1")
    total = memory_gb(model, workload, overhead)
    prefill = prefill_seconds(model, gpu, workload.prompt_tokens, gpus)
    tpot = tpot_seconds(model, gpu, workload, gpus)
    e2e = prefill + max(workload.generated_tokens - 1, 0) * tpot
    return Result(
        gpus=gpus,
        weights_gb=weights_gb(model),
        kv_cache_gb=kv_gb(model, workload.context_tokens, workload.concurrent_requests),
        memory_gb=total,
        memory_per_gpu_gb=total / gpus,
        fits=total <= gpus * gpu.memory_gb,
        ttft_s=prefill,
        tpot_ms=tpot * 1e3,
        e2e_s=e2e,
    )


def recommend(
    model: Model,
    gpu: GPU,
    workload: Workload,
    overhead: float = DEFAULT_OVERHEAD,
) -> Result:
    """Наименьшее число GPU, на которое всё помещается, и скорость на нём."""
    n = gpus_for_memory(model, gpu, workload, overhead)
    return evaluate(model, gpu, workload, n, overhead)
