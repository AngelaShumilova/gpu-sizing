"""Формулы аналитического минимума. Только стандартная библиотека.

Все объёмы в десятичных гигабайтах (10⁹ байт), как в паспортах GPU.
"""

from __future__ import annotations

from dataclasses import dataclass

GB = 1e9

# Байт на параметр (и на одно значение KV-кэша).
BYTES_PER_PARAM: dict[str, float] = {
    "fp32": 4.0,
    "fp16": 2.0,
    "bf16": 2.0,
    "fp8": 1.0,
    "int8": 1.0,
    "fp6": 0.75,
    "fp4": 0.5,
    "nf4": 0.5,
    "int4": 0.5,
    "int2": 0.25,
}

EIGHT_BIT = {"fp8", "int8"}


@dataclass(frozen=True)
class Model:
    name: str
    params_b: float  # все параметры, млрд: определяют память
    n_layers: int
    n_kv_heads: int  # MHA: = числу голов запроса, GQA: меньше, MQA: 1
    head_dim: int
    weights_precision: str = "bf16"
    kv_precision: str = "bf16"
    kv_tensors: int = 2  # 2 — отдельные K и V, 1 — K и V хранятся одним тензором (DeepSeek-V4)
    active_params_b: float | None = None  # для MoE: параметры, работающие на один токен

    @property
    def active_b(self) -> float:
        return self.params_b if self.active_params_b is None else self.active_params_b


@dataclass(frozen=True)
class GPU:
    name: str
    memory_gb: float
    bandwidth_gbs: float  # пропускная способность памяти, ГБ/с
    tflops_fp16: float  # плотный пик FP16/BF16
    tflops_fp8: float | None = None  # плотный пик FP8; None — FP8 не поддерживается


@dataclass(frozen=True)
class Workload:
    prompt_tokens: int
    response_tokens: int
    concurrent_requests: int
    reasoning_tokens: int = 0  # скрытые токены рассуждений: тоже генерируются и лежат в KV

    @property
    def generated_tokens(self) -> int:
        return self.reasoning_tokens + self.response_tokens

    @property
    def context_tokens(self) -> int:
        return self.prompt_tokens + self.generated_tokens


# ── Память ───────────────────────────────────────────────────────────────────


def weights_gb(model: Model) -> float:
    """Веса = параметры × байт на параметр."""
    return model.params_b * BYTES_PER_PARAM[model.weights_precision]


def kv_bytes_per_token(model: Model) -> float:
    """KV на токен = тензоров KV · слои · KV-головы · размер головы · байт.

    Обычно тензоров два: ключ и значение. Если K и V — один тензор, то один.
    """
    return model.kv_tensors * model.n_layers * model.n_kv_heads * model.head_dim * BYTES_PER_PARAM[model.kv_precision]


def kv_gb(model: Model, tokens: int, requests: int) -> float:
    """KV-кэш = KV на токен × токенов в запросе × число запросов."""
    return kv_bytes_per_token(model) * tokens * requests / GB


def memory_gb(model: Model, workload: Workload, overhead: float) -> float:
    """(Веса + KV-кэш всех запросов) × запас фреймворка."""
    kv = kv_gb(model, workload.context_tokens, workload.concurrent_requests)
    return (weights_gb(model) + kv) * overhead


# ── Скорость ─────────────────────────────────────────────────────────────────


def compute_tflops(model: Model, gpu: GPU) -> float:
    """8-битные веса считаются на FP8, если GPU его умеет. Всё остальное — на FP16."""
    if model.weights_precision in EIGHT_BIT and gpu.tflops_fp8:
        return gpu.tflops_fp8
    return gpu.tflops_fp16


def prefill_seconds(model: Model, gpu: GPU, prompt_tokens: int, n_gpu: int) -> float:
    """Prefill упирается в вычисления: 2 FLOP на активный параметр на токен промпта.

    Считаем, что n_gpu делят работу поровну.
    """
    flops = 2 * model.active_b * 1e9 * prompt_tokens
    return flops / (n_gpu * compute_tflops(model, gpu) * 1e12)


def tpot_seconds(model: Model, gpu: GPU, workload: Workload, n_gpu: int) -> float:
    """Decode упирается в память: каждый шаг читает активные веса и KV-кэш всех запросов.

    KV-кэш берём на полный контекст P + R, как и для памяти: это верхняя оценка.
    Считаем, что n_gpu читают память вместе, то есть их пропускная способность складывается.
    """
    weights_bytes = model.active_b * 1e9 * BYTES_PER_PARAM[model.weights_precision]
    kv_bytes = kv_bytes_per_token(model) * workload.context_tokens * workload.concurrent_requests
    return (weights_bytes + kv_bytes) / (n_gpu * gpu.bandwidth_gbs * GB)
