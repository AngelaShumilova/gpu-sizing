# llm-gpu-sizing

Материалы к докладу «Тернистый путь к подсчёту стоимости токена в железе».

Ссылка на запись доклада
https://youtu.be/eb52Xf7tK3o?is=bdcp8IErPgNtKbCc

Сколько видеокарт нужно, чтобы развернуть LLM, и как быстро она будет работать.
Здесь собраны калькулятор, литература, на которой он построен, и аналитический минимум, чтобы пересчитать всё вручную под свою модель.

## Структура

```
calculator/            простой GPU-калькулятор: формулы, HTTP API и страница
analytical-minimum/    формулы с объяснениями и примерами из доклада
literature/            источники по блокам доклада
  01-gpus.md             видеокарты: память против вычислений, паспорта, цены
  02-neural-networks.md  нейросети до LLM: параметры vs FLOPs, активации, DNNMem
  03-llm.md              LLM: веса, KV-кэш, MQA/GQA, MoE, квантизация
  04-inference.md        инференс: prefill/decode, batching, KV-кэш, speculative, reasoning, параллелизм
  05-calculator-validation.md   заготовка: сверка с рецептами vLLM / SGLang
```

## Быстрый старт

```bash
uv sync && uv run fastapi dev src/gpu_calculator/api.py
```

Страница откроется на http://127.0.0.1:8000
