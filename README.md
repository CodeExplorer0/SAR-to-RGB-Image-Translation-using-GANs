# SAR-to-RGB Image Translation using GANs

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org)

A Generative Adversarial Network (GAN) implementation for translating Synthetic Aperture Radar (SAR) images to RGB visual representations.

![Sample Results](samples/Screenshot_2025-02-03_155828.png)

## Features

- U-Net based Generator architecture with skip connections
- PatchGAN Discriminator for detailed local evaluation
- Custom Chromatic Aberration Loss combining:
  - Perceptual loss using VGG19 features
  - Spatial consistency loss
  - Edge-aware loss
- Mixed-precision training support
- GPU acceleration with CUDA

## Installation

1. Clone the repository:
