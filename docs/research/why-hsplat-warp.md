# Why hsplat-warp

`hsplat-warp` is not just a renamed fork. The goal is to make the project easier to extend as a research codebase.

## Fork motivation

Compared with upstream-oriented reproduction code, this fork emphasizes:

- clearer project identity
- more maintainable custom-kernel development
- a lower barrier for future contributors working on accelerated Gaussian paths

## Why Warp matters here

The current project already has a narrow but important custom CUDA hot path for Gaussian `naive_fast`.

Adding [NVIDIA Warp](https://github.com/NVIDIA/warp) as an optional backend helps because:

- the performance-critical custom kernel remains explicit
- backend experimentation stays close to Python
- future researchers can add or modify kernels without immediately dropping into PyTorch C++ extension code

## What is intentionally unchanged

This fork does not try to replace the entire runtime stack with Warp.

It intentionally keeps:

- PyTorch for orchestration and FFT-heavy operations
- the native CUDA extension as a compatible path
- the upstream-friendly repository layout

That balance is the main reason the project branding can evolve to `hsplat-warp` without forcing a risky package rename at the same time.
