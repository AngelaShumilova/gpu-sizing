"""HTTP API и страница калькулятора.

    uv run fastapi dev src/gpu_calculator/api.py
"""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import catalog
from .formulas import Workload
from .sizing import DEFAULT_OVERHEAD, recommend

STATIC = Path(__file__).parent / "static"

app = FastAPI(title="GPU calculator")


class SizingRequest(BaseModel):
    model: str
    gpu: str
    precision: Literal["fp32", "fp16", "bf16", "fp8", "int8", "fp6", "fp4", "nf4", "int4", "int2"] | None = None
    prompt_tokens: int = Field(ge=1)
    response_tokens: int = Field(ge=0)
    reasoning_tokens: int = Field(default=0, ge=0)
    concurrent_requests: int = Field(ge=1)
    overhead: float = Field(default=DEFAULT_OVERHEAD, ge=1.0)


def _inputs(req: SizingRequest):
    try:
        model, gpu = catalog.model(req.model), catalog.gpu(req.gpu)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e.args[0])) from None
    if req.precision:
        model = replace(model, weights_precision=req.precision)
    workload = Workload(req.prompt_tokens, req.response_tokens, req.concurrent_requests, req.reasoning_tokens)
    return model, gpu, workload


@app.get("/api/models")
def list_models() -> list[dict]:
    return [asdict(m) for m in catalog.models().values()]


@app.get("/api/gpus")
def list_gpus() -> list[dict]:
    return [asdict(g) for g in catalog.gpus().values()]


@app.post("/api/recommend")
def recommend_layout(req: SizingRequest) -> dict:
    model, gpu, workload = _inputs(req)
    return asdict(recommend(model, gpu, workload, req.overhead))


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")
