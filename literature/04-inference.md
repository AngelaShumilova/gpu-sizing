# 4. Инференс

Особенности фреймворка (batching, PagedAttention, prefix caching, спекулятивный декодинг, reasoning) не меняют базовые формулы, но сдвигают реальный результат, поэтому одной аналитики недостаточно и нужен замер.

## Batching и планировщик

- Yu et al. *Orca: A Distributed Serving System for Transformer-Based Generative Models.* OSDI 2022. <https://www.usenix.org/system/files/osdi22-yu.pdf>
- Agrawal et al. *Sarathi-Serve.* OSDI 2024. Chunked prefill. <https://arxiv.org/abs/2403.02310>
- Anyscale. *How continuous batching enables 23x throughput.* <https://www.anyscale.com/blog/continuous-batching-llm-inference>

## KV-кэш: PagedAttention и prefix caching

- Kwon et al. *Efficient Memory Management for LLM Serving with PagedAttention.* SOSP 2023. <https://arxiv.org/abs/2309.06180>
- Блог vLLM: <https://vllm.ai/blog/2023-06-20-vllm> 
- Zheng et al. *SGLang: Efficient Execution of Structured Language Model Programs (RadixAttention).* <https://arxiv.org/abs/2312.07104> 
- Блог LMSYS: <https://www.lmsys.org/blog/2024-01-17-sglang/> 

## Спекулятивный декодинг

- Leviathan, Kalman, Matias. *Fast Inference from Transformers via Speculative Decoding.* ICML 2023. <https://arxiv.org/abs/2211.17192> 
- Chen et al. *Accelerating Large Language Model Decoding with Speculative Sampling.* 2023. <https://arxiv.org/abs/2302.01318>
- Li et al. *EAGLE-3.* 2025. <https://arxiv.org/abs/2503.01840>
- 
## Reasoning и tool calling

- Epoch AI. 2025. <https://epoch.ai/data-insights/output-length>
- Independent analysis of AI. <https://artificialanalysis.ai/models/comparisons?compare=claude-opus-5-5%2Cclaude-opus-5-5-medium>

## Параллелизм

- Shoeybi et al. *Megatron-LM.* 2020. <https://arxiv.org/abs/1909.08053> 
- Narayanan et al. *Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM.* SC 2021. TP+PP+DP. <https://arxiv.org/abs/2104.04473> 
- Lepikhin et al. *GShard.* 2020. <https://arxiv.org/abs/2006.16668>
- Jacobs et al. *DeepSpeed Ulysses.* 2023. Sequence/context parallelism. <https://arxiv.org/abs/2309.14509>

## Экономика токена

- Crusoe. *Tokenomics in the age of agentic inference.* 2026. <https://www.crusoe.ai/resources/blog/tokenomics-in-the-age-of-agentic-inference>
- Cisco. *Agent tokenomics.* <https://www.cisco.com/site/us/en/learn/topics/artificial-intelligence/agent-tokenomics.html>
- Spheron. *Agentic AI inference cost 2026.* <https://spheron.network/blog/agentic-ai-inference-cost-2026>
