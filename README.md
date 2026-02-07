# Performance Benchmarks: DINOv3 & CIFAR-100 CNN

This repo contains simple scripts to benchmark the performance for the following on Ryzen 7 8745HS

- **Training:** Lightweight CIFAR-100 CNN
- **Inference:** [DINOv3 ViT-S/16 distilled (21M)](https://github.com/facebookresearch/dinov3)

## Train
- **CPU:** AMD Ryzen 7 8745HS
- **iGPU:** AMD Radeon 780M with ROCm 7.x

```
Device cpu speed: 1353.8 samples/sec
Device cuda speed: 3018.0 samples/sec
GPU is 2.23x speed of CPU
```
