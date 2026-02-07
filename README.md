# Performance Benchmarks: DINOv3 & CIFAR-100 CNN

This repo contains simple scripts to benchmark the performance on

- **CPU:** AMD Ryzen 7 8745HS
- **iGPU:** AMD Radeon 780M with ROCm 7.x support

## Train

- **Model:** Lightweight CIFAR-100 CNN
```bash
Device cpu speed: 1353.8 samples/sec
Device cuda speed: 3018.0 samples/sec
GPU is 2.23x speed of CPU
```

## Inference

- **Model:** [DINOv3 ViT-S/16 distilled (21M)](https://github.com/facebookresearch/dinov3)

Flash Attention support on ROCm is currently experimental. To enable it, use the flag `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`
```bash
TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 python profiler.py

CPU total time: 20.102ms
CUDA total time: 10.819ms
```
