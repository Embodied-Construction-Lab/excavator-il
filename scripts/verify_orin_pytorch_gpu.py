#!/usr/bin/env python3
"""Verify the CUDA kernels needed before installing the ACT runtime."""

from __future__ import annotations

import json
import time

import torch
from torch import nn


def _timed_cuda(operation, *, repeats: int) -> float:
    for _ in range(3):
        operation()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(repeats):
        operation()
    torch.cuda.synchronize()
    return (time.perf_counter() - started) * 1000.0 / repeats


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the PyTorch container")

    device = torch.device("cuda")
    results: dict[str, object] = {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
    }

    for dtype, label in ((torch.float32, "fp32"), (torch.float16, "fp16")):
        left = torch.randn((512, 512), device=device, dtype=dtype)
        right = torch.randn((512, 512), device=device, dtype=dtype)
        output = left @ right
        results[f"matmul_{label}_finite"] = bool(torch.isfinite(output).all().item())
        results[f"matmul_{label}_mean_ms"] = _timed_cuda(
            lambda: torch.mm(left, right), repeats=20
        )

    convolution = nn.Conv2d(3, 64, 7, stride=2, padding=3).to(device).half().eval()
    image = torch.randn((1, 3, 480, 640), device=device, dtype=torch.float16)
    with torch.inference_mode():
        convolution_output = convolution(image)
        results["conv_fp16_mean_ms"] = _timed_cuda(
            lambda: convolution(image), repeats=20
        )
    results["conv_fp16_shape"] = list(convolution_output.shape)
    results["conv_fp16_finite"] = bool(
        torch.isfinite(convolution_output).all().item()
    )
    results["allocated_mb"] = round(torch.cuda.memory_allocated() / 1024**2, 3)
    results["reserved_mb"] = round(torch.cuda.memory_reserved() / 1024**2, 3)

    finite_checks = [
        value for key, value in results.items() if key.endswith("_finite")
    ]
    results["passed"] = all(finite_checks)
    print(json.dumps(results, indent=2))
    return 0 if results["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
