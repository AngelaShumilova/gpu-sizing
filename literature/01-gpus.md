# 1. Видеокарты

Какие параметры ускорителя важны для сайзинга (объём памяти, скорость чтения памяти, пиковые FLOPS), почему вычисления растут быстрее памяти, сколько это стоит.

## Память против вычислений

- Williams, Waterman, Patterson. *Roofline: An Insightful Visual Performance Model for Multicore Architectures.* CACM 52(4), 2008. <https://people.eecs.berkeley.edu/~kubitron/cs252/handouts/papers/RooflineVyNoYellow.pdf>
- Gholami et al. *AI and Memory Wall.* IEEE Micro, 2024. <https://arxiv.org/abs/2403.14123>
- Hobbhahn, Heim, Aydos. *Trends in Machine Learning Hardware.* Epoch AI, 2023. <https://epoch.ai/publications/trends-in-machine-learning-hardware>
- Sevilla et al. *Compute Trends Across Three Eras of Machine Learning.* IJCNN 2022. <https://arxiv.org/abs/2202.05924>
- Hooker. *The Hardware Lottery.* CACM 64(12), 2021. <https://arxiv.org/abs/2009.06489>
- NVIDIA. *Matrix Multiplication Background User's Guide.* 2023. <https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html>

## Паспортные характеристики

- NVIDIA V100: <https://images.nvidia.com/content/technologies/volta/pdf/tesla-volta-v100-datasheet.pdf>
- NVIDIA A100: <https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-nvidia-us-2188504-web.pdf>
- NVIDIA H100: <https://resources.nvidia.com/en-us-hopper-architecture/nvidia-h100-tensor-c>
- NVIDIA H200: <https://www.nvidia.com/en-us/data-center/h200/>
- NVIDIA L40S: <https://www.nvidia.com/en-us/data-center/l40s/>
- NVIDIA Blackwell / HGX B200: <https://resources.nvidia.com/en-us-blackwell-architecture>
- NVIDIA GB200 NVL72: <https://www.nvidia.com/en-us/data-center/gb200-nvl72/>
- NVIDIA Rubin. <https://en.wikipedia.org/wiki/Rubin_(microarchitecture)>
- AMD Instinct MI300X: <https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-data-sheet.pdf>
- Google TPU7x (Ironwood): <https://docs.cloud.google.com/tpu/docs/tpu7x>
- MLCommons. *MLPerf Inference*, datacenter. <https://mlcommons.org/benchmarks/inference-datacenter/>

## Цены и рынок

- CloudZero. *H100 GPU cost.* <https://www.cloudzero.com/blog/h100-gpu-cost/>
- IntuitionLabs. *NVIDIA AI GPU pricing guide.* <https://intuitionlabs.ai/articles/nvidia-ai-gpu-pricing-guide>
- getdeploying. *H100 cloud pricing.* <https://getdeploying.com/gpus/nvidia-h100>
- SemiAnalysis InferenceX. *Llama-3.3-70B, B200 vs H200 per dollar.* <https://inferencex.semianalysis.com/compare-per-dollar/llama-3-3-70b-b200-vs-h200>
